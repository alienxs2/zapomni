#!/usr/bin/env python3
"""
Basic Zapomni Usage Examples

This script demonstrates fundamental operations with the Zapomni memory system.
"""

import asyncio
from datetime import datetime
from zapomni_mcp.client import ZapomniClient


async def example_1_simple_memory():
    """Example 1: Add and search a simple memory"""
    print("\n=== Example 1: Simple Memory Addition ===\n")

    # Initialize client
    client = ZapomniClient()
    await client.connect()

    try:
        # Add a simple memory
        print("Adding memory: 'Python was created by Guido van Rossum in 1991'")
        result = await client.add_memory(
            content="Python was created by Guido van Rossum in 1991",
            metadata={"category": "programming_history", "source": "manual"}
        )
        print(f"✓ Memory stored with ID: {result['memory_id']}")
        print(f"  Chunks created: {result['chunks_count']}")
        print(f"  Entities extracted: {result['entities_count']}")

        # Search for the memory
        print("\nSearching for: 'Who created Python?'")
        search_results = await client.search_memory(
            query="Who created Python?",
            limit=5
        )

        for i, result in enumerate(search_results, 1):
            print(f"\n{i}. {result['content']}")
            print(f"   Score: {result['score']:.2%}")
            print(f"   Memory ID: {result['memory_id']}")

    finally:
        await client.disconnect()


async def example_2_batch_memories():
    """Example 2: Add multiple memories in batch"""
    print("\n=== Example 2: Batch Memory Addition ===\n")

    client = ZapomniClient()
    await client.connect()

    try:
        # Create a list of memories
        memories = [
            "JavaScript was created by Brendan Eich in 1995",
            "Rust was introduced by Mozilla in 2010",
            "Go was created by Google in 2007",
            "TypeScript was developed by Microsoft in 2012",
        ]

        print(f"Adding {len(memories)} memories in batch...\n")

        for memory in memories:
            result = await client.add_memory(
                content=memory,
                metadata={"source": "batch_example"}
            )
            print(f"✓ {memory[:50]}...")

        # Get statistics
        print("\nMemory Statistics:")
        stats = await client.get_stats()
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")

    finally:
        await client.disconnect()


async def example_3_semantic_search():
    """Example 3: Semantic search with various query types"""
    print("\n=== Example 3: Semantic Search Variations ===\n")

    client = ZapomniClient()
    await client.connect()

    try:
        # Add some context memories
        context_memories = [
            "Python is used for data science and web development",
            "JavaScript runs in web browsers and Node.js",
            "Rust is used for systems programming",
            "Go excels at concurrent programming",
        ]

        print("Setting up memories...")
        for memory in context_memories:
            await client.add_memory(content=memory)

        # Different search queries
        queries = [
            "programming paradigms",
            "web development",
            "systems programming",
            "concurrent programming",
        ]

        print("\nSearching with different queries:\n")

        for query in queries:
            print(f"Query: '{query}'")
            results = await client.search_memory(query=query, limit=3)

            for result in results[:2]:  # Show top 2
                print(f"  - {result['content'][:60]}... (score: {result['score']:.1%})")

            print()

    finally:
        await client.disconnect()


async def example_4_entity_relationships():
    """Example 4: Extract and explore entity relationships"""
    print("\n=== Example 4: Entity Relationships ===\n")

    client = ZapomniClient()
    await client.connect()

    try:
        # Add a memory with multiple entities
        memory_content = """
        Alice is a software engineer at Google.
        She specializes in distributed systems.
        Google was founded by Larry Page and Sergey Brin.
        They met at Stanford University.
        """

        print("Adding memory with relationships...")
        result = await client.add_memory(
            content=memory_content,
            metadata={"type": "biographical"}
        )

        print(f"✓ Memory added")
        print(f"  Entities extracted: {result['entities_count']}")
        print(f"  Relationships detected: {result['relationships_count']}")

        # Get detailed entity information
        print("\nEntity Information:")
        if 'entities' in result:
            for entity in result['entities'][:5]:
                print(f"  - {entity['name']} ({entity['type']})")

    finally:
        await client.disconnect()


async def example_5_contextual_search():
    """Example 5: Search with context window"""
    print("\n=== Example 5: Contextual Search ===\n")

    client = ZapomniClient()
    await client.connect()

    try:
        # Add related memories to build context
        memories = [
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are inspired by biological neurons",
            "Deep learning uses multiple layers of neural networks",
            "Transformers revolutionized NLP with attention mechanisms",
            "GPT models are large language models based on transformers",
        ]

        print("Building knowledge base...")
        for memory in memories:
            await client.add_memory(content=memory)

        # Search with context
        print("\nSearching: 'How do modern language models work?'\n")

        results = await client.search_memory(
            query="How do modern language models work?",
            limit=5,
            include_context=True
        )

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['content']}")
            print(f"   Relevance: {result['score']:.1%}")

            if 'context' in result:
                print(f"   Context: {result['context']}")

            print()

    finally:
        await client.disconnect()


async def example_6_metadata_filtering():
    """Example 6: Add and search with metadata"""
    print("\n=== Example 6: Metadata Filtering ===\n")

    client = ZapomniClient()
    await client.connect()

    try:
        # Add memories with different metadata
        memories_with_metadata = [
            ("Python uses indentation", {"language": "Python", "topic": "syntax"}),
            ("JavaScript uses curly braces", {"language": "JavaScript", "topic": "syntax"}),
            ("Rust uses ownership system", {"language": "Rust", "topic": "memory"}),
            ("Go has goroutines", {"language": "Go", "topic": "concurrency"}),
        ]

        print("Adding memories with metadata...\n")

        for content, metadata in memories_with_metadata:
            await client.add_memory(content=content, metadata=metadata)
            print(f"✓ {content}")

        # Search with metadata filter
        print("\nSearching for Python-related content:\n")

        results = await client.search_memory(
            query="programming syntax",
            filter={"language": "Python"},
            limit=5
        )

        for result in results:
            print(f"- {result['content']}")
            print(f"  Metadata: {result.get('metadata', {})}")

    finally:
        await client.disconnect()


async def main():
    """Run all examples"""
    examples = [
        example_1_simple_memory,
        example_2_batch_memories,
        example_3_semantic_search,
        example_4_entity_relationships,
        example_5_contextual_search,
        example_6_metadata_filtering,
    ]

    print("=" * 60)
    print("Zapomni Basic Usage Examples")
    print("=" * 60)

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
