# Zapomni Project - Project Manager Handoff

**Last Updated**: 2025-11-28 (Session #12 - PERFORMANCE FIX)
**Project Status**: v0.4.0 Foundation COMPLETE | BUG-007 FIXED | 6 bugs remaining
**Version**: v0.3.1 (released) | v0.4.0 Foundation (merged)
**Branch**: `main`

---

## START HERE (Новый PM)

### Что было сделано в Session #12:

1. **FIXED: BUG-007 (Issue #13) - Performance 7-45x медленнее**
   - Добавлен Ollama batch API `/api/embed` в `ollama_embedder.py`
   - Интегрирован EmbeddingCache (Redis + in-memory fallback) в `memory_processor.py`
   - Включён semantic cache по умолчанию
   - **Ожидаемое улучшение: -60-90% latency**

### Предыдущие сессии:
- **Session #11**: Найдено 7 багов, создан roadmap v0.5-v1.0, 19 issues
- **Session #10**: v0.4.0 Foundation complete, Tree-sitter 41 languages

### Текущее состояние:
```
Branch: main
Unit tests: 2089 passed, 11 skipped
E2E tests: 88 passed, 1 xfailed
Tree-sitter: Foundation COMPLETE (41 languages, 221 tests)
Known bugs: 6 (2 critical/high, 4 medium/low) - BUG-007 FIXED!
```

### Шаг 1: Проверь состояние
```bash
cd /home/dev/zapomni
git pull origin main
source .venv/bin/activate
make test                     # Unit: ~2089 passed
gh issue list --milestone "v0.5.0 - Solid Foundation"
```

### Шаг 2: Прочитай отчёты
```
/home/dev/zapomi_anal/
├── ZAPOMNI_PRODUCT_ANALYSIS_REPORT.md   # Полный анализ + конкуренты
├── ZAPOMNI_ROADMAP_POST_BUGFIX.md       # Детальный roadmap v0.5-v1.0
├── test_results_summary.json            # 7 багов с root cause
└── issues/                              # Детальное описание каждого бага
```

---

## ТЕКУЩИЕ ПРИОРИТЕТЫ

### ✅ FIXED - Session #12

| Issue | Bug | Severity | Описание | Status |
|-------|-----|----------|----------|--------|
| #13 | BUG-007 | HIGH | **Performance 7-45x медленнее** | ✅ FIXED |

**Что было сделано:**
- `ollama_embedder.py:359-470` - Новый метод `_call_ollama_batch()` для batch API
- `memory_processor.py:1142-1218` - Интеграция кэша в `_generate_embeddings()`
- `__main__.py:240-328` - Создание Redis client и EmbeddingCache
- `config.py:257` - `enable_semantic_cache = True` по умолчанию

### P0 - Критические баги (СНАЧАЛА ЭТО!)

| Issue | Bug | Severity | Описание | Estimate |
|-------|-----|----------|----------|----------|
| #12 | BUG-005 | CRITICAL | Workspace isolation сломана | 4-8 часов |
| #14 | BUG-002 | HIGH | Code indexing не работает | 3-5 часов |

### P1 - Важные баги

| Issue | Bug | Severity | Описание | Estimate |
|-------|-----|----------|----------|----------|
| #15 | BUG-003 | MEDIUM | AST = 0 functions/classes | 2-4 часа |
| #16 | BUG-004 | HIGH | Workspace state не сохраняется | 2-4 часа |

### P2-P3 - Могут подождать

| Issue | Bug | Severity | Estimate |
|-------|-----|----------|----------|
| #17 | BUG-001 | MEDIUM | 30 мин |
| #18 | BUG-006 | LOW | 2-4 часа |

---

## ROADMAP v0.5 → v1.0

### GitHub Milestones (уже созданы)

| # | Milestone | Issues | Duration |
|---|-----------|--------|----------|
| 1 | [v0.5.0 - Solid Foundation](https://github.com/alienxs2/zapomni/milestone/1) | #19, #20, #21 | 3-4 нед |
| 2 | [v0.6.0 - Code Intelligence](https://github.com/alienxs2/zapomni/milestone/2) | #22, #23, #24 | 4-5 нед |
| 3 | [v0.7.0 - Search Excellence](https://github.com/alienxs2/zapomni/milestone/3) | #25, #26 | 3-4 нед |
| 4 | [v0.8.0 - Knowledge Graph 2.0](https://github.com/alienxs2/zapomni/milestone/4) | #27 | 4-5 нед |
| 5 | [v0.9.0 - Scale & Performance](https://github.com/alienxs2/zapomni/milestone/5) | #28 | 3-4 нед |
| 6 | [v1.0.0 - Production Ready](https://github.com/alienxs2/zapomni/milestone/6) | #29, #30 | 4-5 нед |

**Общий срок до v1.0: ~5-6 месяцев**

---

## КОНКУРЕНТНОЕ ПОЗИЦИОНИРОВАНИЕ

### Zapomni vs Топ конкуренты

| Feature | Zapomni | Mem0 | Zep | Cognee |
|---------|---------|------|-----|--------|
| Code Intelligence | **41 lang** | No | No | ~10 lang |
| 100% Local | **Yes** | No | No | Partial |
| Tree-sitter AST | **Yes** | No | No | No |
| Knowledge Graph | Yes | Yes | Yes | Yes |
| Temporal Model | No | No | **Yes** | No |
| Stars | ~100 | 43.6k | 23.8k | 9.3k |

### Уникальное позиционирование

> **"The AI memory that truly understands your code"**
> - 41 language AST support (уникально!)
> - 100% local-first (privacy)
> - Tree-sitter deep parsing

---

## ПОРЯДОК РАБОТЫ

### Рекомендуемый порядок после Session #12:

1. **Fix #12 (Workspace isolation)** - CRITICAL, основная функциональность
   - Inject mcp_server в tools
   - resolve_workspace_id() когда не указан
   - Files: `server.py`, `add_memory.py`, `search_memory.py`

2. **Fix #14, #15 (Code indexing)** - Наш differentiator
   - Integrate Tree-sitter
   - Real functions_found/classes_found
   - Files: `index_codebase.py`

3. **Fix #16 (Workspace state)** - Related to #12
   - Instance-level state for stdio mode
   - Files: `workspace_tools.py`, `server.py`

4. **Start v0.5.0 issues** (#19, #20, #21)
   - PythonExtractor
   - TypeScriptExtractor
   - Full Tree-sitter integration

---

## БЫСТРЫЕ КОМАНДЫ

```bash
# Тесты
make test                          # Unit тесты
make e2e                           # E2E тесты

# GitHub
gh issue list                      # Все issues
gh issue list --state open         # Открытые issues
gh issue view 12                   # Посмотреть issue #12

# Сервер
make docker-up                     # FalkorDB + Redis
make server                        # MCP сервер

# Redis для кэша (опционально, есть in-memory fallback)
REDIS_ENABLED=true docker compose up -d redis
```

---

## ВАЖНЫЕ ФАЙЛЫ

### Код
```
/home/dev/zapomni/src/
├── zapomni_core/
│   ├── embeddings/
│   │   ├── ollama_embedder.py     # Batch API (Session #12)
│   │   └── embedding_cache.py     # Cache implementation
│   ├── memory_processor.py        # Cache integration (Session #12)
│   └── treesitter/                # Tree-sitter модуль (ГОТОВ)
├── zapomni_mcp/
│   ├── __main__.py                # Redis init (Session #12)
│   └── tools/                     # MCP tools (17 штук)
└── tests/                         # 2089+ тестов
```

### Документация
```
/home/dev/zapomni/
├── CHANGELOG.md                   # История изменений
├── README.md                      # Основная документация
└── .claude/resume-prompt.md       # ЭТА ИНСТРУКЦИЯ

/home/dev/zapomi_anal/
├── ZAPOMNI_PRODUCT_ANALYSIS_REPORT.md
├── ZAPOMNI_ROADMAP_POST_BUGFIX.md
├── test_results_summary.json
└── issues/BUG-*.md
```

---

## ИСТОРИЯ СЕССИЙ

### Session 2025-11-28 #12 (PERFORMANCE FIX)
**PM**: AI Assistant (Claude Opus 4.5)

**Выполнено**:
- ✅ FIXED: Issue #13 (BUG-007) - Performance 7-45x медленнее
- Добавлен Ollama batch API `/api/embed`
- Интегрирован EmbeddingCache в memory_processor
- Включён semantic cache по умолчанию
- 2089 unit tests passed

**Коммиты**:
- `81c63087` - fix(performance): Implement embedding caching and batch API
- `d16c0c0a` - docs: Update resume-prompt.md

### Session 2025-11-28 #11 (STRATEGIC PLANNING)
**PM**: AI Assistant (Claude Opus 4.5)

**Выполнено**:
- Глубокое тестирование: найдено 7 багов
- Исследование 43+ конкурентов
- Создание Product Analysis Report
- Создание Roadmap v0.5-v1.0
- 7 bug issues (#12-18)
- 6 milestones
- 12 feature issues (#19-30)

**Артефакты**:
- `/home/dev/zapomi_anal/ZAPOMNI_PRODUCT_ANALYSIS_REPORT.md`
- `/home/dev/zapomi_anal/ZAPOMNI_ROADMAP_POST_BUGFIX.md`

### Session 2025-11-28 #10 (Previous)
- v0.4.0 Foundation complete
- 221 Tree-sitter tests
- PR #7 merged

---

## КОНТАКТЫ

- **Repository**: https://github.com/alienxs2/zapomni
- **Issues**: https://github.com/alienxs2/zapomni/issues
- **Milestones**: https://github.com/alienxs2/zapomni/milestones
- **Owner**: Goncharenko Anton (alienxs2)

---

## СЛЕДУЮЩИЕ ШАГИ ДЛЯ НОВОГО PM

1. **Прочитать эту инструкцию полностью**
2. **Запустить тесты**: `make test` - убедиться что всё работает
3. **Начать с Issue #12 (Workspace isolation)** - самый критичный баг
4. **Использовать Task tool с opus** для сложных задач

**Успех = Fix 6 bugs → v0.5.0 → v0.6.0 → ... → v1.0 Launch**
