"""Safe materialization-time ``adata.obs`` filter parsing and evaluation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class ObsFilterError(ValueError):
    """Raised when an obs filter expression is invalid or unsupported."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: Any
    position: int


@dataclass(frozen=True)
class _Comparison:
    column: str
    operator: str
    value: Any = None


@dataclass(frozen=True)
class _Logical:
    operator: str
    left: Any
    right: Any


_KEYWORDS = {
    "and": "AND",
    "or": "OR",
    "is": "IS",
    "not": "NOT",
    "in": "IN",
    "null": "NULL",
    "true": "BOOL",
    "false": "BOOL",
}


def filter_obs_rows(obs: pd.DataFrame, expression: str | None) -> np.ndarray:
    """Return source row indices retained by a safe obs filter expression."""
    if expression is None:
        return np.arange(len(obs), dtype=np.int64)

    text = expression.strip()
    if not text:
        return np.arange(len(obs), dtype=np.int64)

    tokens = _tokenize(text)
    parser = _Parser(tokens)
    ast_root = parser.parse()
    mask = _evaluate(ast_root, obs)
    return np.flatnonzero(mask)


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if text.startswith("==", i):
            tokens.append(_Token("OP", "==", i))
            i += 2
            continue
        if text.startswith("!=", i):
            tokens.append(_Token("OP", "!=", i))
            i += 2
            continue
        if text.startswith(">=", i):
            tokens.append(_Token("OP", ">=", i))
            i += 2
            continue
        if text.startswith("<=", i):
            tokens.append(_Token("OP", "<=", i))
            i += 2
            continue
        if ch in "><(),[]":
            tokens.append(_Token(ch, ch, i))
            i += 1
            continue
        if ch in {'"', "'"}:
            j = i + 1
            escaped = False
            while j < n:
                if escaped:
                    escaped = False
                elif text[j] == "\\":
                    escaped = True
                elif text[j] == ch:
                    break
                j += 1
            if j >= n:
                raise ObsFilterError(f"unterminated string starting at position {i}")
            literal = ast.literal_eval(text[i : j + 1])
            tokens.append(_Token("VALUE", literal, i))
            i = j + 1
            continue
        if ch == "`":
            j = text.find("`", i + 1)
            if j == -1:
                raise ObsFilterError(f"unterminated column name starting at position {i}")
            name = text[i + 1 : j]
            if not name:
                raise ObsFilterError("empty backtick-quoted column name is not allowed")
            tokens.append(_Token("IDENT", name, i))
            i = j + 1
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n and text[j] in "0123456789._eE+-":
                if text[j].isspace():
                    break
                j += 1
            raw = text[i:j]
            try:
                value = float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError as exc:
                raise ObsFilterError(
                    f"invalid numeric literal {raw!r} at position {i}"
                ) from exc
            tokens.append(_Token("VALUE", value, i))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            raw = text[i:j]
            lowered = raw.lower()
            keyword = _KEYWORDS.get(lowered)
            if keyword == "BOOL":
                tokens.append(_Token("VALUE", lowered == "true", i))
            elif keyword is not None:
                tokens.append(_Token(keyword, lowered, i))
            else:
                tokens.append(_Token("IDENT", raw, i))
            i = j
            continue
        raise ObsFilterError(f"unexpected character {ch!r} at position {i}")
    tokens.append(_Token("EOF", None, n))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]):
        self._tokens = tokens
        self._index = 0

    def parse(self) -> Any:
        expr = self._parse_or()
        if self._peek().kind != "EOF":
            token = self._peek()
            raise ObsFilterError(
                f"unexpected token {token.value!r} at position {token.position}"
            )
        return expr

    def _parse_or(self) -> Any:
        expr = self._parse_and()
        while self._match("OR"):
            expr = _Logical("or", expr, self._parse_and())
        return expr

    def _parse_and(self) -> Any:
        expr = self._parse_primary()
        while self._match("AND"):
            expr = _Logical("and", expr, self._parse_primary())
        return expr

    def _parse_primary(self) -> Any:
        if self._match("("):
            expr = self._parse_or()
            self._expect(")")
            return expr
        return self._parse_comparison()

    def _parse_comparison(self) -> _Comparison:
        column = self._expect("IDENT").value
        if self._match("IS"):
            negate = self._match("NOT")
            self._expect("NULL")
            return _Comparison(column, "is_not_null" if negate else "is_null")
        if self._match("NOT"):
            self._expect("IN")
            return _Comparison(column, "not_in", self._parse_list())
        if self._match("IN"):
            return _Comparison(column, "in", self._parse_list())
        if self._peek().kind == "OP":
            operator = self._advance().value
            value = self._expect("VALUE").value
            if value is None:
                raise ObsFilterError("use 'is null' or 'is not null' for null checks")
            return _Comparison(column, operator, value)
        token = self._peek()
        raise ObsFilterError(
            f"expected a comparison operator after column {column!r} at position {token.position}"
        )

    def _parse_list(self) -> list[Any]:
        self._expect("[")
        values: list[Any] = []
        if self._match("]"):
            return values
        while True:
            values.append(self._expect("VALUE").value)
            if self._match("]"):
                return values
            self._expect(",")

    def _peek(self) -> _Token:
        return self._tokens[self._index]

    def _advance(self) -> _Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _match(self, kind: str) -> bool:
        if self._peek().kind == kind:
            self._index += 1
            return True
        return False

    def _expect(self, kind: str) -> _Token:
        token = self._peek()
        if token.kind != kind:
            raise ObsFilterError(
                f"expected {kind} at position {token.position}, found {token.kind}"
            )
        self._index += 1
        return token


def _evaluate(node: Any, obs: pd.DataFrame) -> np.ndarray:
    if isinstance(node, _Logical):
        left = _evaluate(node.left, obs)
        right = _evaluate(node.right, obs)
        return left & right if node.operator == "and" else left | right

    if not isinstance(node, _Comparison):
        raise ObsFilterError(f"unsupported filter node: {type(node).__name__}")

    if node.column not in obs.columns:
        raise ObsFilterError(f"obs filter references unknown column {node.column!r}")

    series = obs[node.column]
    try:
        if node.operator == "is_null":
            mask = series.isna()
        elif node.operator == "is_not_null":
            mask = ~series.isna()
        elif node.operator == "in":
            mask = series.isin(node.value)
        elif node.operator == "not_in":
            mask = ~series.isin(node.value)
        elif node.operator == "==":
            mask = series == node.value
        elif node.operator == "!=":
            mask = series != node.value
        elif node.operator == ">":
            mask = series > node.value
        elif node.operator == ">=":
            mask = series >= node.value
        elif node.operator == "<":
            mask = series < node.value
        elif node.operator == "<=":
            mask = series <= node.value
        else:
            raise ObsFilterError(f"unsupported operator {node.operator!r}")
    except Exception as exc:
        raise ObsFilterError(
            f"obs filter comparison failed for column {node.column!r}: {exc}"
        ) from exc

    if isinstance(mask, np.ndarray):
        return mask.astype(bool, copy=False)
    if hasattr(mask, "fillna"):
        mask = mask.fillna(False)
    return mask.to_numpy(dtype=bool)
