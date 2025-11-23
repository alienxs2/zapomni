# Spec Creation Prompt Template

**Type:** Agent Prompt Template
**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Last Updated:** 2025-11-22

---

ЗАДАЧА: Создать спецификацию {level}-level для {component_name}

КОНТЕКСТ ПРОЕКТА:
- Проект: Zapomni - local-first MCP memory system
- Author: Goncharenko Anton aka alienxs2
- License: MIT
- GitHub: https://github.com/alienxs2/zapomni

STEERING DOCUMENTS (прочитать обязательно):
1. /home/dev/zapomni/.spec-workflow/steering/product.md
2. /home/dev/zapomni/.spec-workflow/steering/tech.md
3. /home/dev/zapomni/.spec-workflow/steering/structure.md

{if level > 1}
PARENT SPECIFICATIONS (прочитать обязательно):
{parent_spec_files}
{endif}

МЕТОДОЛОГИЯ:
Прочитай: /home/dev/zapomni/SPEC_METHODOLOGY.md
Используй соответствующий template для {level}-level spec.

СОЗДАТЬ ФАЙЛ: `.spec-workflow/specs/{level}/{component_name}.md`

ТРЕБОВАНИЯ:

1. **Completeness:**
   - Все секции template заполнены
   - {if level==1: API interfaces defined}
   - {if level==2: All public methods documented}
   - {if level==3: All edge cases (min 3) + test scenarios (min 5)}

2. **Consistency:**
   - Terminology совпадает с parent specs
   - Alignment со steering documents
   - Data types consistent

3. **Quality:**
   - Examples provided
   - Design decisions explained
   - Non-functional requirements specified

ФОРМАТ ОТЧЁТА:

## SPEC CREATED

**File:** `.spec-workflow/specs/{level}/{component_name}.md`
**Lines:** [count] lines
**Size:** [words] words

**Sections Completed:**
- ✅ Overview
- ✅ {level-specific sections}
- ✅ References

**Key Decisions:**
1. {decision 1}
2. {decision 2}

**Dependencies Identified:**
- {dependency 1}
- {dependency 2}

**Ready for Verification:** Yes

---

НАЧИНАЙ СОЗДАНИЕ СПЕЦИФИКАЦИИ.
