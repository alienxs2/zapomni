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
└── __main__.py     # Инициализация всех компонентов

zapomni_core/       # Бизнес-логика
├── memory_processor.py    # Главный оркестратор
├── chunking/              # SemanticChunker
├── embeddings/            # OllamaEmbedder
├── extractors/            # EntityExtractor (SpaCy NER)
└── search/                # VectorSearch, HybridSearch

zapomni_db/         # Database layer
├── falkordb_client.py     # FalkorDB операции
├── cypher_query_builder.py # Генерация Cypher запросов
├── schema_manager.py      # Индексы и схема
└── models.py              # Pydantic модели
```

## Текущий статус (2025-11-25)
- **MCP подключается:** ✅
- **add_memory:** ✅ работает (с тегами и source)
- **search_memory:** ✅ работает (с фильтрами)
- **build_graph:** ✅ исправлен (EntityExtractor инициализирован)
- **get_stats, graph_status, export_graph, delete_memory:** ✅

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
5. **Пользователь перезапускает** Claude Code для подхвата изменений

## Важные особенности FalkorDB
- **Vector search:** `db.idx.vector.queryNodes(label, attribute, k, vecf32(vector))`
- **Возвращает DISTANCE** (0=идентично), не similarity — нужна конверсия
- **CREATE INDEX:** без имени! `CREATE INDEX FOR (n:Label) ON (n.prop)`
- **Порт:** 6381 (не стандартный 6379)

## Полезные команды
```bash
# Проверить FalkorDB
docker exec zapomni_falkordb redis-cli -p 6379 GRAPH.QUERY zapomni_memory "MATCH (m:Memory) RETURN count(m)"

# Запустить MCP сервер вручную
cd /home/dev/zapomni && .venv/bin/python -m zapomni_mcp

# Тест поиска напрямую
.venv/bin/python -c "
from falkordb import FalkorDB
db = FalkorDB(host='localhost', port=6381)
graph = db.select_graph('zapomni_memory')
result = graph.query('MATCH (m:Memory) RETURN m.id, m.tags LIMIT 5')
print(result.result_set)
"
```

## Последние коммиты (2025-11-25)
- `791ee000` — SpaCy model для EntityExtractor
- `a4a11c5b` — EntityExtractor + tags/source storage
- `16be5b44` — Schema init + SearchResult mapping
- `749f2b2b` — Distance→Similarity конверсия
- `2a4dc493` — queryNodes 4 аргумента

## Известные ограничения
- **Graph Health: Warning** — нормально без entities
- **FalkorDB CREATE VECTOR INDEX warning** — обрабатывается try/except
- **Editable install** — изменения применяются сразу, но нужен перезапуск MCP

## Начало работы
```
Продолжаем отладку проекта Zapomni MCP (/home/dev/zapomni).
Прочитай .claude/resume-prompt.md для контекста.
```
