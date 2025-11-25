# Zapomni MCP - Resume Prompt

## Проект
**Zapomni** — local-first MCP сервер памяти для AI агентов.
- **Репозиторий:** https://github.com/alienxs2/zapomni
- **Путь:** `/home/dev/zapomni`
- **Стек:** Python 3.12, FalkorDB (граф + векторы), Ollama (embeddings), SpaCy (NER)

## Архитектура
```
zapomni_mcp/        # MCP сервер (entry point: python -m zapomni_mcp)
├── server.py       # MCPServer class
├── tools/          # add_memory, search_memory, build_graph, etc.
└── __main__.py     # Инициализация (SpaCy НЕ грузится здесь!)

zapomni_core/       # Бизнес-логика
├── memory_processor.py    # Главный оркестратор + LAZY LOADING
├── chunking/              # SemanticChunker
├── embeddings/            # OllamaEmbedder
├── extractors/            # EntityExtractor (SpaCy NER)
├── graph/                 # GraphBuilder
└── search/                # VectorSearch, HybridSearch

zapomni_db/         # Database layer
├── falkordb_client.py     # FalkorDB операции
├── cypher_query_builder.py # Генерация Cypher запросов
├── schema_manager.py      # Индексы и схема
└── models.py              # Pydantic модели
```

## Текущий статус (2025-11-25)
- **MCP подключается:** ✅ быстро (~0.3 сек благодаря lazy loading)
- **add_memory:** ✅ работает (без загрузки SpaCy)
- **search_memory:** ✅ работает (без загрузки SpaCy)
- **build_graph:** ✅ работает (SpaCy грузится лениво при первом вызове)
- **get_stats, graph_status, export_graph, delete_memory:** ✅
- **get_related, clear_all:** ✅

## Ключевое: Ленивая загрузка SpaCy
```python
# memory_processor.py использует @property для lazy loading:

@property
def extractor(self):
    if self._extractor is None:
        # SpaCy загружается ТОЛЬКО здесь, при первом доступе
        spacy_model = spacy.load("en_core_web_sm")
        self._extractor = EntityExtractor(spacy_model=spacy_model)
    return self._extractor

@property
def graph_builder(self):
    if self._graph_builder is None:
        self._graph_builder = GraphBuilder(
            entity_extractor=self.extractor,  # триггерит загрузку SpaCy
            db_client=self.db_client,
        )
    return self._graph_builder
```

**Важно:** В коде проверять `self._extractor` (не `self.extractor`!) чтобы не триггерить загрузку.

## Конфигурация
```bash
# ~/.claude.json — MCP сервер
"zapomni": {
  "command": "/home/dev/zapomni/.venv/bin/python",
  "args": ["-m", "zapomni_mcp"],
  "env": {
    "FALKORDB_HOST": "localhost",
    "FALKORDB_PORT": "6381",
    "OLLAMA_BASE_URL": "http://localhost:11434"
  }
}
```

## Docker сервисы
```bash
docker ps | grep -E "falkor|ollama"
# zapomni_falkordb — порт 6381
# ollama — порт 11434
```

## Workflow разработки
1. **Редактирую** файлы через Edit/Write
2. **Тестирую** через Bash (`.venv/bin/python -c "..."`) или MCP tools
3. **Коммичу** с описательным сообщением + Co-Authored-By
4. **Пушу** в GitHub (`git push origin main`)
5. **Пользователь перезапускает** Claude Code для подхвата изменений MCP

## Стиль коммитов
```
type(scope): краткое описание

- детали изменения 1
- детали изменения 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```
Типы: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`

## Важные особенности FalkorDB
- **Vector search:** `db.idx.vector.queryNodes(label, attribute, k, vecf32(vector))`
- **Возвращает DISTANCE** (0=идентично), не similarity — нужна конверсия
- **CREATE INDEX:** без имени! `CREATE INDEX FOR (n:Label) ON (n.prop)`
- **Порт:** 6381 (не стандартный 6379)

## Полезные команды
```bash
# Проверить Docker сервисы
docker ps | grep -E "falkor|ollama"

# Проверить FalkorDB
docker exec zapomni_falkordb redis-cli -p 6379 GRAPH.QUERY zapomni_memory "MATCH (n) RETURN labels(n)[0], count(n)"

# Запустить MCP сервер вручную
cd /home/dev/zapomni && .venv/bin/python -m zapomni_mcp

# Тест компонентов напрямую
.venv/bin/python -c "
from zapomni_core.logging_service import LoggingService
LoggingService.configure_logging(level='WARNING')
# ... твой код
"

# Проверить статус git
git status && git log --oneline -5
```

## Последние коммиты
- `cfc84cb0` — **perf: Lazy loading для SpaCy/EntityExtractor**
- `e7fbb332` — docs: Resume prompt
- `791ee000` — fix: SpaCy model для EntityExtractor
- `a4a11c5b` — fix: EntityExtractor + tags/source storage
- `16be5b44` — fix: Schema init + SearchResult mapping

## Известные особенности
- **Lazy loading** — SpaCy грузится только при build_graph, не при старте
- **FalkorDB SHOW INDEXES** — не поддерживается, используем try/except
- **Editable install** — изменения применяются сразу, но нужен перезапуск MCP

## MCP Tools (все работают)
| Tool | Описание | Загружает SpaCy? |
|------|----------|------------------|
| `add_memory` | Добавить память | Нет |
| `search_memory` | Поиск по памяти | Нет |
| `build_graph` | Построить граф знаний | Да (лениво) |
| `get_stats` | Статистика системы | Нет |
| `graph_status` | Статус графа | Нет |
| `export_graph` | Экспорт графа | Нет |
| `delete_memory` | Удалить память | Нет |
| `get_related` | Связанные сущности | Нет |
| `clear_all` | Очистить всё | Нет |

## Начало работы
```
Продолжаем работу над Zapomni MCP (/home/dev/zapomni).
Прочитай .claude/resume-prompt.md для контекста.
```
