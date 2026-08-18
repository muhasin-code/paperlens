# Retrieval Quality Benchmark

## Evaluation Setup

- **Dataset size**: 26 queries
- **Retrieval strategies**: Baseline Semantic, Hybrid RRF, Hybrid + Reranker, Semantic + Reranker
- **Metrics**: Precision@5, Recall@5, Hit@5, MRR, Paper-Precision@5, Paper-Hit@5
- **Environment**: CPU-only (Dell Latitude 7490, Intel UHD 620, 7.6 GB RAM)
- **Mode**: Headless (no Ollama/LLM dependencies)

## Comparison Tables

### Precision and Recall (chunk-level)

| Metric | Baseline | Hybrid (RRF) | Hybrid + Reranker | Semantic + Reranker |
|--------|----------|--------------|-------------------|---------------------|
| precision_at_5 | 0.423 | 0.515 (+21.8%) | 0.308 (-27.3%) | 0.231 (-45.5%) |
| recall_at_5 | 0.212 | 0.258 (+21.8%) | 0.154 (-27.3%) | 0.115 (-45.5%) |
| hit_at_5 | 0.885 | 0.962 (+8.7%) | 0.846 (-4.3%) | 0.692 (-21.7%) |
| mrr | 0.683 | 0.749 (+9.8%) | 0.608 (-10.9%) | 0.468 (-31.5%) |

### Paper-level metrics

| Metric | Baseline | Hybrid (RRF) | Hybrid + Reranker | Semantic + Reranker |
|--------|----------|--------------|-------------------|---------------------|
| paper_precision_at_5 | 0.592 | 0.692 (+16.9%) | 0.469 (-20.8%) | 0.369 (-37.7%) |
| paper_hit_at_5 | 0.962 | 1.000 (+4.0%) | 0.923 (-4.0%) | 0.769 (-20.0%) |

## Methodology

### Strategies Evaluated

1. **Baseline Semantic**: `SemanticRetriever` top-5 cosine search over ChromaDB.
2. **Hybrid (RRF)**: `HybridRetriever` merges semantic + BM25 with Reciprocal Rank Fusion.
3. **Hybrid + Reranker**: Hybrid top-20 candidates reranked to top-5.
4. **Semantic + Reranker**: Semantic top-50 candidates reranked to top-5.

### Gold Dataset

Each query includes `relevant_chunk_ids` (5-10 manually curated labels) and `relevant_arxiv_ids` for paper-level scoring. Labels were expanded using both semantic and BM25 discovery from seed papers to reduce single-retriever bias.

### Metric Definitions

- **Precision@5**: |relevant chunks ∩ top-5| / 5
- **Recall@5**: |relevant chunks ∩ top-5| / |all relevant chunks|
- **Hit@5**: 1 if any relevant chunk appears in top-5, else 0
- **MRR**: reciprocal rank of the first relevant chunk in top-5
- **Paper-Precision@5**: fraction of top-5 chunks from a relevant paper
- **Paper-Hit@5**: 1 if any top-5 chunk comes from a relevant paper

## Detailed Results

### Query: How do recent papers approach learning rate scheduling?...

- **Relevant papers**: 2606.23188v1, 2606.22984v2, 2606.24102v1, 2606.23286v1, 2606.23208v1, 2606.22068v2, 2606.23758v1, 2606.22369v1, 2606.23637v1
- **Relevant chunks**: 2606.23188v1_chunk_0021, 2606.22984v2_chunk_0050, 2606.22984v2_chunk_0049, 2606.24102v1_chunk_0048, 2606.23286v1_chunk_0013, 2606.23208v1_chunk_0019, 2606.22068v2_chunk_0001, 2606.23758v1_chunk_0019, 2606.22369v1_chunk_0008, 2606.23637v1_chunk_0025
- **baseline**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.400
- **hybrid**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.400
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.400

### Query: What are attention mechanisms in transformer models?...

- **Relevant papers**: 2606.22406v2, 2606.22430v1, 2606.21564v1, 2606.21821v1, 2606.23940v1, 2606.21344v1, 2606.21364v1, 2606.24696v1
- **Relevant chunks**: 2606.22406v2_chunk_0004, 2606.22406v2_chunk_0007, 2606.22430v1_chunk_0038, 2606.22430v1_chunk_0037, 2606.21564v1_chunk_0006, 2606.21821v1_chunk_0038, 2606.23940v1_chunk_0012, 2606.21344v1_chunk_0005, 2606.21364v1_chunk_0008, 2606.24696v1_chunk_0008
- **baseline**: P@5=1.000, R@5=0.500, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid**: P@5=1.000, R@5=0.500, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid_reranker**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **semantic_reranker**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=1.000

### Query: What is the difference between Adam and SGD optimizers?...

- **Relevant papers**: 2606.21433v1, 2606.22326v1, 2606.23286v1
- **Relevant chunks**: 2606.21433v1_chunk_0002, 2606.21433v1_chunk_0003, 2606.21433v1_chunk_0010, 2606.21433v1_chunk_0012, 2606.21433v1_chunk_0011, 2606.21433v1_chunk_0013, 2606.21433v1_chunk_0014, 2606.22326v1_chunk_0001, 2606.22326v1_chunk_0003, 2606.23286v1_chunk_0012
- **baseline**: P@5=1.000, R@5=0.500, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid**: P@5=1.000, R@5=0.500, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.400

### Query: CNN architecture for image classification...

- **Relevant papers**: 2606.22400v1, 2606.22889v1, 2606.21289v1
- **Relevant chunks**: 2606.22400v1_chunk_0014, 2606.22400v1_chunk_0016, 2606.22400v1_chunk_0015, 2606.22400v1_chunk_0022, 2606.22400v1_chunk_0004, 2606.22400v1_chunk_0001, 2606.22889v1_chunk_0016, 2606.22889v1_chunk_0010, 2606.22889v1_chunk_0012, 2606.21289v1_chunk_0017
- **baseline**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **hybrid**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **semantic_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.600

### Query: What is reinforcement learning?...

- **Relevant papers**: 2606.21943v1, 2606.23603v1, 2606.23978v1, 2606.22146v1, 2606.21604v1
- **Relevant chunks**: 2606.21943v1_chunk_0006, 2606.21943v1_chunk_0003, 2606.21943v1_chunk_0031, 2606.21943v1_chunk_0052, 2606.21943v1_chunk_0062, 2606.21943v1_chunk_0066, 2606.23603v1_chunk_0006, 2606.23978v1_chunk_0016, 2606.22146v1_chunk_0042, 2606.21604v1_chunk_0001
- **baseline**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=0.500, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **semantic_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.400

### Query: How does batch normalization improve training?...

- **Relevant papers**: 2606.21916v1, 2606.23880v1, 2606.23758v1, 2606.21957v1, 2606.23617v1, 2606.21297v1, 2606.21712v1
- **Relevant chunks**: 2606.21916v1_chunk_0017, 2606.23880v1_chunk_0008, 2606.23758v1_chunk_0024, 2606.23758v1_chunk_0053, 2606.21957v1_chunk_0018, 2606.23617v1_chunk_0020, 2606.23617v1_chunk_0019, 2606.21297v1_chunk_0012, 2606.21712v1_chunk_0026, 2606.21712v1_chunk_0028
- **baseline**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **hybrid**: P@5=1.000, R@5=0.500, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.200

### Query: What are the benefits of residual connections in deep networ...

- **Relevant papers**: 2606.23477v1, 2606.23364v1, 2606.22112v1, 2606.22369v1, 2606.23856v1
- **Relevant chunks**: 2606.23477v1_chunk_0040, 2606.23477v1_chunk_0039, 2606.23477v1_chunk_0029, 2606.23477v1_chunk_0041, 2606.23477v1_chunk_0046, 2606.23477v1_chunk_0002, 2606.23364v1_chunk_0029, 2606.22112v1_chunk_0016, 2606.22369v1_chunk_0006, 2606.23856v1_chunk_0009
- **baseline**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.400
- **hybrid**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.600
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.400
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.333, Paper-P@5=0.200

### Query: How do transformers handle positional information?...

- **Relevant papers**: 2606.23251v1, 2606.22325v1, 2606.22984v2, 2606.21562v1, 2606.21611v1, 2606.22752v1, 2606.23356v1
- **Relevant chunks**: 2606.23251v1_chunk_0017, 2606.22325v1_chunk_0017, 2606.22984v2_chunk_0024, 2606.22984v2_chunk_0008, 2606.22984v2_chunk_0023, 2606.21562v1_chunk_0001, 2606.21562v1_chunk_0003, 2606.21611v1_chunk_0025, 2606.22752v1_chunk_0013, 2606.23356v1_chunk_0033
- **baseline**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid**: P@5=1.000, R@5=0.500, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.333, Paper-P@5=0.800
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.333, Paper-P@5=0.400

### Query: What is the role of dropout in neural network regularization...

- **Relevant papers**: 2606.23942v1, 2606.22917v1, 2606.21427v1, 2606.21593v1, 2606.21795v1
- **Relevant chunks**: 2606.23942v1_chunk_0004, 2606.23942v1_chunk_0005, 2606.23942v1_chunk_0011, 2606.23942v1_chunk_0001, 2606.23942v1_chunk_0002, 2606.23942v1_chunk_0013, 2606.22917v1_chunk_0004, 2606.21427v1_chunk_0019, 2606.21593v1_chunk_0037, 2606.21795v1_chunk_0015
- **baseline**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **hybrid**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=1.000
- **semantic_reranker**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=1.000

### Query: How does gradient clipping prevent exploding gradients?...

- **Relevant papers**: 2606.22466v1, 2606.22436v1, 2606.21943v1, 2606.22669v1, 2606.22932v1, 2606.23932v1
- **Relevant chunks**: 2606.22466v1_chunk_0003, 2606.22436v1_chunk_0031, 2606.22436v1_chunk_0038, 2606.22436v1_chunk_0035, 2606.21943v1_chunk_0048, 2606.22669v1_chunk_0032, 2606.22669v1_chunk_0019, 2606.22466v1_chunk_0002, 2606.22932v1_chunk_0004, 2606.23932v1_chunk_0006
- **baseline**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.400
- **hybrid**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **semantic_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=1.000

### Query: What are the advantages of layer normalization over batch no...

- **Relevant papers**: 2606.21916v1, 2606.21590v1, 2606.21847v1, 2606.23880v1, 2606.23942v1, 2606.23364v1, 2606.23607v1, 2606.21868v1, 2606.22369v1
- **Relevant chunks**: 2606.21916v1_chunk_0017, 2606.21590v1_chunk_0012, 2606.21847v1_chunk_0027, 2606.23880v1_chunk_0008, 2606.23942v1_chunk_0008, 2606.23364v1_chunk_0017, 2606.23607v1_chunk_0005, 2606.21868v1_chunk_0017, 2606.21590v1_chunk_0011, 2606.22369v1_chunk_0008
- **baseline**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **hybrid**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.200

### Query: How do autoencoders learn compressed representations?...

- **Relevant papers**: 2606.22994v1, 2606.23964v1, 2606.24046v1, 2606.23356v1, 2606.22352v1, 2606.22889v1
- **Relevant chunks**: 2606.22994v1_chunk_0001, 2606.22994v1_chunk_0019, 2606.22994v1_chunk_0020, 2606.23964v1_chunk_0004, 2606.23964v1_chunk_0018, 2606.23964v1_chunk_0017, 2606.24046v1_chunk_0002, 2606.23356v1_chunk_0023, 2606.22352v1_chunk_0021, 2606.22889v1_chunk_0010
- **baseline**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000
- **hybrid**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.200
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.200, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.200

### Query: What is transfer learning in computer vision?...

- **Relevant papers**: 2606.22400v1, 2606.23286v1, 2606.21289v1, 2606.24601v1, 2606.22383v1, 2606.23851v1
- **Relevant chunks**: 2606.22400v1_chunk_0001, 2606.22400v1_chunk_0011, 2606.22400v1_chunk_0016, 2606.22400v1_chunk_0022, 2606.23286v1_chunk_0009, 2606.21289v1_chunk_0009, 2606.24601v1_chunk_0027, 2606.22383v1_chunk_0024, 2606.22400v1_chunk_0026, 2606.23851v1_chunk_0009
- **baseline**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.400
- **hybrid**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.250, Paper-P@5=0.600
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.400
- **semantic_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.400

### Query: How does label smoothing improve model calibration?...

- **Relevant papers**: 2606.21513v1, 2606.21422v1, 2606.22917v1, 2606.23213v1, 2606.22200v1, 2606.23155v1, 2606.22775v2, 2606.22991v1
- **Relevant chunks**: 2606.21513v1_chunk_0028, 2606.21422v1_chunk_0025, 2606.21513v1_chunk_0031, 2606.22917v1_chunk_0029, 2606.23213v1_chunk_0011, 2606.22200v1_chunk_0012, 2606.22200v1_chunk_0013, 2606.23155v1_chunk_0022, 2606.22775v2_chunk_0019, 2606.22991v1_chunk_0003
- **baseline**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=1.000
- **hybrid**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.333, Paper-P@5=0.400
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000

### Query: What are the key components of a GAN architecture?...

- **Relevant papers**: 2606.21289v1, 2606.22700v1, 2606.23856v1, 2606.22304v1, 2606.24087v1, 2606.21513v1
- **Relevant chunks**: 2606.21289v1_chunk_0007, 2606.22700v1_chunk_0010, 2606.22700v1_chunk_0018, 2606.22700v1_chunk_0021, 2606.21289v1_chunk_0028, 2606.23856v1_chunk_0008, 2606.22304v1_chunk_0002, 2606.24087v1_chunk_0008, 2606.21513v1_chunk_0038, 2606.21289v1_chunk_0021
- **baseline**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.400

### Query: How does knowledge distillation transfer knowledge between m...

- **Relevant papers**: 2606.21994v1, 2606.24143v1, 2606.23356v1, 2606.24601v1, 2606.22600v2, 2606.23276v1
- **Relevant chunks**: 2606.21994v1_chunk_0001, 2606.21994v1_chunk_0019, 2606.21994v1_chunk_0020, 2606.24143v1_chunk_0026, 2606.24143v1_chunk_0027, 2606.23356v1_chunk_0010, 2606.24601v1_chunk_0004, 2606.22600v2_chunk_0018, 2606.22600v2_chunk_0019, 2606.23276v1_chunk_0004
- **baseline**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.400
- **hybrid**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.333, Paper-P@5=0.400
- **hybrid_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000

### Query: What is the difference between supervised and self-supervise...

- **Relevant papers**: 2606.22177v1, 2606.21590v1, 2606.21260v1, 2606.24263v1
- **Relevant chunks**: 2606.22177v1_chunk_0001, 2606.22177v1_chunk_0006, 2606.22177v1_chunk_0039, 2606.21590v1_chunk_0003, 2606.21590v1_chunk_0005, 2606.21590v1_chunk_0015, 2606.21590v1_chunk_0001, 2606.21590v1_chunk_0010, 2606.21260v1_chunk_0081, 2606.24263v1_chunk_0018
- **baseline**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.600
- **hybrid**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.600
- **hybrid_reranker**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=0.500, Paper-P@5=0.800
- **semantic_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.600

### Query: How do diffusion models generate high-quality images?...

- **Relevant papers**: 2606.23627v1, 2606.22314v2, 2606.22521v1, 2606.22239v1, 2606.22837v1
- **Relevant chunks**: 2606.23627v1_chunk_0003, 2606.23627v1_chunk_0019, 2606.23627v1_chunk_0020, 2606.23627v1_chunk_0022, 2606.22314v2_chunk_0008, 2606.22314v2_chunk_0012, 2606.22314v2_chunk_0020, 2606.22521v1_chunk_0001, 2606.22239v1_chunk_0011, 2606.22837v1_chunk_0001
- **baseline**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.250, Paper-P@5=0.200
- **hybrid**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.250, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000

### Query: What are the challenges of training very deep neural network...

- **Relevant papers**: 2606.23477v1, 2606.23364v1, 2606.23742v1
- **Relevant chunks**: 2606.23477v1_chunk_0002, 2606.23477v1_chunk_0005, 2606.23477v1_chunk_0029, 2606.23477v1_chunk_0032, 2606.23477v1_chunk_0034, 2606.23477v1_chunk_0035, 2606.23477v1_chunk_0036, 2606.23477v1_chunk_0038, 2606.23364v1_chunk_0029, 2606.23742v1_chunk_0038
- **baseline**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.400
- **hybrid**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.200
- **hybrid_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.200
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.400

### Query: How does contrastive learning learn representations without ...

- **Relevant papers**: 2606.21838v1, 2606.23570v1, 2606.21957v1
- **Relevant chunks**: 2606.21838v1_chunk_0001, 2606.21838v1_chunk_0004, 2606.21838v1_chunk_0011, 2606.21838v1_chunk_0010, 2606.21838v1_chunk_0002, 2606.23570v1_chunk_0001, 2606.23570v1_chunk_0003, 2606.23570v1_chunk_0010, 2606.23570v1_chunk_0014, 2606.21957v1_chunk_0004
- **baseline**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.600
- **hybrid**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.600
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.500, Paper-P@5=0.400

### Query: What is the purpose of weight decay in optimization?...

- **Relevant papers**: 2606.23357v1, 2606.23637v1, 2606.23662v1, 2606.21593v1, 2606.21496v1, 2606.22874v1, 2606.21690v2, 2606.23942v1
- **Relevant chunks**: 2606.23357v1_chunk_0005, 2606.23637v1_chunk_0008, 2606.23662v1_chunk_0081, 2606.23662v1_chunk_0079, 2606.23637v1_chunk_0012, 2606.21593v1_chunk_0033, 2606.21496v1_chunk_0025, 2606.22874v1_chunk_0031, 2606.21690v2_chunk_0029, 2606.23942v1_chunk_0004
- **baseline**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.600
- **hybrid**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.200, Paper-P@5=1.000
- **hybrid_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.200, Paper-P@5=0.400
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000

### Query: How do normalization layers affect gradient flow?...

- **Relevant papers**: 2606.23364v1, 2606.22752v1, 2606.23942v1, 2606.21590v1, 2606.22874v1, 2606.23637v1, 2606.24011v1, 2606.22084v1, 2606.24790v1
- **Relevant chunks**: 2606.23364v1_chunk_0018, 2606.22752v1_chunk_0014, 2606.23942v1_chunk_0004, 2606.21590v1_chunk_0021, 2606.22874v1_chunk_0031, 2606.23637v1_chunk_0007, 2606.24011v1_chunk_0032, 2606.22084v1_chunk_0040, 2606.24790v1_chunk_0007, 2606.24790v1_chunk_0018
- **baseline**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.200
- **hybrid**: P@5=0.800, R@5=0.400, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.600
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000

### Query: What are the benefits of mixed precision training?...

- **Relevant papers**: 2606.21498v1, 2606.23546v1, 2606.22874v1, 2606.22984v2, 2606.24102v1, 2606.23155v1, 2606.21718v1, 2606.23637v1, 2606.23370v1, 2606.22600v2
- **Relevant chunks**: 2606.21498v1_chunk_0011, 2606.23546v1_chunk_0004, 2606.22874v1_chunk_0031, 2606.22984v2_chunk_0049, 2606.24102v1_chunk_0048, 2606.23155v1_chunk_0066, 2606.21718v1_chunk_0006, 2606.23637v1_chunk_0025, 2606.23370v1_chunk_0018, 2606.22600v2_chunk_0018
- **baseline**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.250, Paper-P@5=0.200
- **hybrid**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=0.200, Paper-P@5=0.200
- **hybrid_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000
- **semantic_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000

### Query: How does early stopping prevent overfitting?...

- **Relevant papers**: 2606.22377v1, 2606.23601v1, 2606.22053v1, 2606.22389v1, 2606.21917v1, 2606.22167v1, 2606.21230v1, 2606.21364v1
- **Relevant chunks**: 2606.22377v1_chunk_0022, 2606.22377v1_chunk_0021, 2606.23601v1_chunk_0005, 2606.23601v1_chunk_0008, 2606.22053v1_chunk_0065, 2606.22389v1_chunk_0013, 2606.21917v1_chunk_0003, 2606.22167v1_chunk_0009, 2606.21230v1_chunk_0004, 2606.21364v1_chunk_0011
- **baseline**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.800

### Query: What is the role of the learning rate in optimization?...

- **Relevant papers**: 2606.23364v1, 2606.21683v1, 2606.23286v1, 2606.23208v1, 2606.22053v1, 2606.23758v1, 2606.21593v1, 2606.23188v1, 2606.22326v1, 2606.22639v1
- **Relevant chunks**: 2606.23364v1_chunk_0028, 2606.21683v1_chunk_0006, 2606.23286v1_chunk_0013, 2606.23208v1_chunk_0019, 2606.22053v1_chunk_0057, 2606.23758v1_chunk_0019, 2606.21593v1_chunk_0033, 2606.23188v1_chunk_0021, 2606.22326v1_chunk_0010, 2606.22639v1_chunk_0036
- **baseline**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **hybrid**: P@5=0.600, R@5=0.300, Hit@5=1, MRR=1.000, Paper-P@5=0.800
- **hybrid_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **semantic_reranker**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.500, Paper-P@5=0.400

### Query: How do ensemble methods improve predictive performance?...

- **Relevant papers**: 2606.21929v1, 2606.24047v1, 2606.22026v1, 2606.24062v1, 2606.24263v1
- **Relevant chunks**: 2606.21929v1_chunk_0001, 2606.21929v1_chunk_0002, 2606.21929v1_chunk_0019, 2606.24047v1_chunk_0001, 2606.24047v1_chunk_0005, 2606.24047v1_chunk_0010, 2606.22026v1_chunk_0062, 2606.22026v1_chunk_0102, 2606.24062v1_chunk_0014, 2606.24263v1_chunk_0015
- **baseline**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=0.250, Paper-P@5=0.800
- **hybrid**: P@5=0.400, R@5=0.200, Hit@5=1, MRR=1.000, Paper-P@5=0.600
- **hybrid_reranker**: P@5=0.000, R@5=0.000, Hit@5=0, MRR=0.000, Paper-P@5=0.000
- **semantic_reranker**: P@5=0.200, R@5=0.100, Hit@5=1, MRR=1.000, Paper-P@5=0.200

## Analysis

### Performance Summary

**Hybrid (RRF)** improves average precision@5 by **+21.8%** relative to baseline semantic retrieval.

**Hybrid + Reranker** reduces average precision@5 by **-27.3%** relative to baseline semantic retrieval.

**Semantic + Reranker** reduces average precision@5 by **-45.5%** relative to baseline semantic retrieval.

Recall@5 and paper-level metrics should be used alongside precision@5 when judging hybrid retrieval, since BM25 can surface relevant papers with different chunk boundaries than the gold labels.

## Limitations

1. **Small test set**: ~25 queries provide directional signal, not exhaustive coverage.
2. **Static corpus**: Results depend on the current arXiv snapshot.
3. **Chunk-level labels**: Adjacent chunks from the same paper may be equally valid.
4. **CPU-only latency**: Latency numbers are indicative, not benchmarked here.

---

## Files

- `eval/gold_dataset.jsonl`: Gold evaluation dataset
- `scripts/validate_gold_dataset.py`: Validates labels against ChromaDB
- `scripts/evaluate_retrieval.py`: Headless evaluation script

*Benchmark generated: 2026-08-18 05:05:27 UTC*
