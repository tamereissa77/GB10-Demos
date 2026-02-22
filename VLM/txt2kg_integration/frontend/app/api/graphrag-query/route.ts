import { NextRequest, NextResponse } from 'next/server';
import { ArangoDBService } from '@/lib/arangodb';
import { EmbeddingsService } from '@/lib/embeddings';
import { QdrantService } from '@/lib/qdrant';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';

/**
 * GraphRAG query endpoint - combines Vector DB (Qdrant) + Graph DB (ArangoDB) + local NIM LLM
 * POST /api/graphrag-query
 */
export async function POST(req: NextRequest) {
    const startTime = Date.now();

    try {
        const body = await req.json();
        const { query, topK = 5 } = body;

        if (!query || typeof query !== 'string') {
            return NextResponse.json({ error: 'Query is required' }, { status: 400 });
        }

        console.log(`\n🔍 GraphRAG query: "${query}"`);

        // ============== STEP 1: Vector search in Qdrant ==============
        let vectorContext = '';
        let vectorChunks: Array<{ text: string; score: number }> = [];

        try {
            const embeddingsService = EmbeddingsService.getInstance();
            await embeddingsService.initialize();

            const qdrantService = QdrantService.getInstance();
            const isRunning = await qdrantService.isQdrantRunning();

            if (isRunning) {
                if (!qdrantService.isInitialized()) {
                    await qdrantService.initialize();
                }

                const queryEmbedding = (await embeddingsService.encode([query]))[0];
                const searchResults = await qdrantService.findSimilarDocuments(queryEmbedding, topK);

                if (searchResults && searchResults.length > 0) {
                    vectorChunks = searchResults.map((r: any) => ({
                        text: r.metadata?.text || r.metadata?.pageContent || '',
                        score: r.score
                    })).filter((c: any) => c.text.length > 0);

                    vectorContext = vectorChunks.map(c => c.text).join('\n\n');
                    console.log(`📄 Vector search: found ${vectorChunks.length} relevant chunks`);
                } else {
                    console.log('📄 Vector search: no results found');
                }
            } else {
                console.log('⚠️ Qdrant is not running, skipping vector search');
            }
        } catch (vectorError) {
            console.warn('⚠️ Vector search error:', vectorError);
        }

        // ============== STEP 2: Graph search in ArangoDB ==============
        let graphTriples: Array<{ subject: string; predicate: string; object: string }> = [];

        try {
            const arangoService = ArangoDBService.getInstance();
            const url = process.env.ARANGODB_URL;
            const dbName = process.env.ARANGODB_DB;
            await arangoService.initialize(url, dbName);

            const queryWords = query.toLowerCase()
                .replace(/[?!.,;:]/g, '')
                .split(/\s+/)
                .filter((w: string) => w.length > 2);

            const entities = await arangoService.executeQuery(
                `FOR e IN entities
          LET nameLower = LOWER(e.name)
          FILTER (
            ${queryWords.map((_: string, i: number) => `CONTAINS(nameLower, @word${i})`).join(' OR ')}
          )
          RETURN e`,
                Object.fromEntries(queryWords.map((w: string, i: number) => [`word${i}`, w]))
            );

            if (entities.length > 0) {
                const entityKeys = entities.map((e: any) => e._key);
                console.log(`🔗 Graph search: found ${entities.length} matching entities`);

                const edges = await arangoService.executeQuery(
                    `FOR e IN relationships
            FILTER e._from IN @froms OR e._to IN @tos
            LET fromEntity = DOCUMENT(e._from)
            LET toEntity = DOCUMENT(e._to)
            RETURN {
              subject: fromEntity.name,
              predicate: e.type,
              object: toEntity.name
            }`,
                    {
                        froms: entityKeys.map((k: string) => `entities/${k}`),
                        tos: entityKeys.map((k: string) => `entities/${k}`)
                    }
                );

                graphTriples = edges.filter((t: any) => t.subject && t.predicate && t.object);
                console.log(`🔗 Graph search: found ${graphTriples.length} related triples`);
            } else {
                console.log('🔗 Graph search: no entity matches, fetching all triples');
                const allEdges = await arangoService.executeQuery(
                    `FOR e IN relationships
            LIMIT 50
            LET fromEntity = DOCUMENT(e._from)
            LET toEntity = DOCUMENT(e._to)
            RETURN {
              subject: fromEntity.name,
              predicate: e.type,
              object: toEntity.name
            }`
                );
                graphTriples = allEdges.filter((t: any) => t.subject && t.predicate && t.object);
                console.log(`🔗 Graph search: retrieved ${graphTriples.length} triples as fallback`);
            }
        } catch (graphError) {
            console.warn('⚠️ Graph search error:', graphError);
        }

        // ============== STEP 3: Generate answer with local NIM LLM ==============
        const nimLlmUrl = process.env.NIM_LLM_URL || 'http://nim-llm:8000/v1';
        const nimLlmModel = process.env.NIM_LLM_MODEL || 'qwen/qwen3-32b-dgx-spark';

        const graphContext = graphTriples.length > 0
            ? graphTriples.map(t => `• ${t.subject} → ${t.predicate} → ${t.object}`).join('\n')
            : 'No graph data available.';

        let answer = '';

        try {
            const llm = new ChatOpenAI({
                modelName: nimLlmModel,
                temperature: 0.2,
                maxTokens: 1024,
                openAIApiKey: 'not-needed',
                configuration: {
                    baseURL: nimLlmUrl,
                    timeout: 180000,
                },
            });

            const promptTemplate = PromptTemplate.fromTemplate(`
You are a helpful financial analyst answering questions about documents. 
Use BOTH sources of information below to give the most accurate answer possible.

**Document Text Context** (semantic search results):
{vectorContext}

**Knowledge Graph Facts** (structured data - exact numbers and relationships):
{graphContext}

Rules:
- Prefer exact numbers and facts from the Knowledge Graph when available
- Use the Document Text Context for explanations, context, and details
- If the knowledge graph has specific numbers, always include them in your answer
- If you cannot find relevant information in either source, say so clearly
- Answer in the same language as the question

Question: {query}

Answer:`);

            const chain = RunnableSequence.from([
                {
                    vectorContext: () => vectorContext || 'No document text available.',
                    graphContext: () => graphContext,
                    query: () => query,
                },
                promptTemplate,
                llm,
                new StringOutputParser(),
            ]);

            answer = await chain.invoke({});
            console.log(`🤖 LLM answer generated (${answer.length} chars)`);
        } catch (llmError) {
            console.error('❌ LLM error:', llmError);
            answer = `Could not generate LLM answer (NIM may still be loading). Raw data:\n\n**Graph Facts:**\n${graphContext}\n\n**Text Context:**\n${vectorContext || 'None found.'}`;
        }

        const duration = ((Date.now() - startTime) / 1000).toFixed(2);
        console.log(`✅ GraphRAG query completed in ${duration}s`);

        return NextResponse.json({
            success: true,
            answer,
            sources: {
                vectorChunks: vectorChunks.slice(0, 3),
                graphTriples: graphTriples.slice(0, 20),
            },
            metadata: {
                vectorResultCount: vectorChunks.length,
                graphTripleCount: graphTriples.length,
                durationSeconds: parseFloat(duration),
                llmModel: nimLlmModel,
            }
        });

    } catch (error) {
        console.error('❌ GraphRAG query error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        return NextResponse.json(
            { error: `GraphRAG query failed: ${errorMessage}` },
            { status: 500 }
        );
    }
}
