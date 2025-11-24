# Requirements Document - Phase 3 MCP Tools

## Introduction

This specification defines Phase 3 MCP tools for the Zapomni local-first memory server. Phase 1 (MVP) delivered core memory operations (add_memory, search_memory, get_stats). Phase 3 adds advanced capabilities:

1. **Code indexing** - Index Git repositories with AST analysis for code comprehension
2. **Memory management** - Delete specific memories and clear all data
3. **Graph export** - Export knowledge graph in multiple formats for visualization and analysis

These tools complete the Zapomni feature set by enabling code repository integration, data lifecycle management, and knowledge graph interoperability.

**Value Proposition:**
- Developers can index their codebases for semantic code search and understanding
- Users gain fine-grained control over their memory data
- Knowledge graphs can be exported for external analysis, backup, and visualization tools

## Alignment with Product Vision

The Zapomni project aims to be a local-first, privacy-focused personal memory system that:
- Stores and retrieves information semantically using vector embeddings
- Builds knowledge graphs from unstructured text and code
- Provides MCP integration for seamless AI assistant workflows

Phase 3 tools directly support these goals:
- **Code indexing** extends semantic understanding to source code repositories
- **Memory management** ensures user control and privacy through data deletion
- **Graph export** enables interoperability with external tools and backup workflows

## Requirements

### Requirement 1: Index Codebase Tool

**User Story:** As a developer, I want to index my Git repository so that I can semantically search my codebase and understand code relationships through AI assistants.

#### Acceptance Criteria

1. WHEN the tool receives a valid repository path THEN the system SHALL index all code files using AST analysis
2. WHEN the tool receives an invalid path THEN the system SHALL return a validation error with clear message
3. WHEN indexing completes successfully THEN the system SHALL return statistics including file count, total lines, and languages detected
4. WHEN indexing encounters file errors THEN the system SHALL log warnings but continue processing other files
5. WHEN indexing a large repository THEN the system SHALL respect file size limits and skip excessively large files
6. IF a repository contains .gitignore patterns THEN the system SHALL respect those patterns during indexing
7. WHEN the tool processes code files THEN the system SHALL extract:
   - Function definitions and signatures
   - Class definitions and methods
   - Import/dependency relationships
   - Code documentation and comments
8. WHEN indexing completes THEN indexed code SHALL be stored in the knowledge graph and searchable via search_memory

#### Technical Constraints

- Maximum file size: 10MB per file (configurable)
- Supported languages: Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, Kotlin, Scala, R
- AST analysis using tree-sitter or language-specific parsers
- Respects .gitignore patterns and default ignore patterns
- Does not follow symlinks by default

### Requirement 2: Delete Memory Tool

**User Story:** As a user, I want to delete specific memories by ID so that I can remove outdated or unwanted information from my knowledge graph.

#### Acceptance Criteria

1. WHEN the tool receives a valid memory UUID THEN the system SHALL delete the memory and all associated chunks
2. WHEN the tool receives an invalid UUID THEN the system SHALL return a validation error
3. WHEN the tool attempts to delete a non-existent memory THEN the system SHALL return a clear "not found" message
4. WHEN deletion completes successfully THEN the system SHALL return confirmation with the deleted memory ID
5. WHEN deletion fails due to database error THEN the system SHALL return an error and maintain data integrity
6. WHEN a memory is deleted THEN all associated graph nodes and relationships SHALL be removed
7. IF a memory has relationships to other memories THEN the system SHALL preserve those other memories but remove the deleted memory's connections

#### Safety Requirements

- Deletion is irreversible - tool documentation must clearly state this
- UUID validation prevents accidental deletion of wrong memories
- Database transactions ensure atomic deletion (all-or-nothing)
- No cascade deletion of related memories (only direct relationships removed)

### Requirement 3: Clear All Memories Tool

**User Story:** As a user, I want to clear all memories from my knowledge graph so that I can start fresh or reset my memory system.

#### Acceptance Criteria

1. WHEN the tool is invoked WITHOUT confirmation THEN the system SHALL return an error requesting explicit confirmation
2. WHEN the tool receives explicit confirmation (confirm=true) THEN the system SHALL delete all nodes and relationships
3. WHEN clearing completes successfully THEN the system SHALL return statistics showing number of nodes/relationships deleted
4. WHEN clearing fails THEN the system SHALL preserve existing data and return error details
5. WHEN clearing is invoked THEN the system SHALL log a warning with timestamp for audit purposes
6. IF the graph is already empty THEN the system SHALL return success with zero deletion count

#### Safety Requirements

- **CRITICAL:** Requires explicit confirmation flag (confirm=true) to prevent accidental data loss
- Tool description must include clear warnings about irreversible data loss
- Operation must be atomic (all-or-nothing)
- Must log all clear_all operations with timestamp and result
- Should display warning message emphasizing destructive nature

### Requirement 4: Export Graph Tool

**User Story:** As a user, I want to export my knowledge graph in standard formats so that I can visualize, analyze, or backup my data using external tools.

#### Acceptance Criteria

1. WHEN the tool receives a valid format parameter THEN the system SHALL export the graph in that format
2. WHEN the tool receives an invalid format THEN the system SHALL return an error listing supported formats
3. WHEN export completes successfully THEN the system SHALL return the exported data as a formatted string or file path
4. WHEN export format is GraphML THEN the system SHALL:
   - Include all nodes with properties as XML attributes
   - Include all edges with relationship types and properties
   - Follow GraphML 1.0 specification
5. WHEN export format is Cytoscape JSON THEN the system SHALL:
   - Include nodes array with data objects
   - Include edges array with source/target references
   - Follow Cytoscape.js JSON format specification
6. WHEN export format is Neo4j format THEN the system SHALL:
   - Generate Cypher CREATE statements for all nodes
   - Generate Cypher CREATE statements for all relationships
   - Include all node properties and labels
7. WHEN exporting a large graph THEN the system SHALL handle memory efficiently and stream data if needed
8. IF the graph is empty THEN the system SHALL return a valid empty structure for the requested format

#### Supported Export Formats

1. **GraphML** - XML-based graph format, widely supported by visualization tools (Gephi, yEd, Cytoscape)
2. **Cytoscape JSON** - JSON format for Cytoscape.js visualization library
3. **Neo4j** - Cypher statements for importing into Neo4j or compatible graph databases
4. **JSON** (optional) - Simple JSON representation with nodes and edges arrays

#### Technical Requirements

- Export generation must be efficient for graphs with 100K+ nodes
- Exported data must be valid according to format specifications
- Must handle special characters and escape properly for XML/JSON/Cypher
- Should include metadata (export timestamp, format version, node/edge counts)

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility Principle**: Each MCP tool file should focus solely on MCP protocol handling
- **Modular Design**: Delegate business logic to existing core components (RepositoryIndexer, FalkorDBClient, GraphExporter)
- **Dependency Management**: Tools depend on core components, not each other
- **Clear Interfaces**: Follow existing MCP tool patterns from Phase 1 (add_memory, search_memory, get_stats)
- **Consistent Error Handling**: Use existing exception types and format errors uniformly

### Performance

- **Code indexing**: Should process 1000 files within 30 seconds on typical developer hardware
- **Delete memory**: Should complete within 500ms for single memory deletion
- **Clear all**: Should complete within 5 seconds for graphs with 100K nodes
- **Export graph**: Should generate exports within 10 seconds for graphs with 50K nodes

### Security

- **Path validation**: Repository paths must be validated to prevent directory traversal attacks
- **UUID validation**: Memory IDs must be validated as proper UUIDs before deletion
- **Confirmation required**: Destructive operations (clear_all) must require explicit confirmation
- **No credential exposure**: Tool responses must not expose database credentials or internal paths

### Reliability

- **Error recovery**: File-level errors during indexing should not abort entire operation
- **Atomic operations**: Delete and clear operations must be atomic (all-or-nothing)
- **Transaction safety**: Database operations must use transactions to ensure consistency
- **Graceful degradation**: If git metadata extraction fails, indexing should continue with basic file info

### Usability

- **Clear error messages**: Validation errors should explain exactly what went wrong and how to fix it
- **Progress feedback**: Long-running operations (indexing) should provide progress indication
- **Helpful documentation**: Tool descriptions should include examples and usage tips
- **Consistent response format**: All tools follow MCP protocol with content array and isError flag
