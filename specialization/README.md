# Specialization Module

This module contains specialized implementations that extend and enhance the baseline RAG pipeline with advanced features for knowledge graph integration, hybrid retrieval strategies, and enhanced embedding techniques.

## 📁 Folder Structure

```
specialization/
├── __init__.py                 # Module initialization
├── main.py                     # Main entry point for specialized pipelines
├── specialization.py           # Legacy entry point (can be refactored)
│
├── config/                     # Specialized configurations
│   ├── __init__.py
│   └── config.py              # Override baseline configs with specialized settings
│
├── data/                       # Specialized data management
│   ├── raw/                   # Specialized raw documents
│   ├── processed/             # Processed outputs
│   ├── db/                    # Vector stores & databases
│   ├── knowledge_graphs/      # Knowledge graph outputs
│   ├── embeddings/            # Embedding outputs
│   └── sqlite/                # SQLite database outputs
│
├── components/                 # Enhanced/specialized components
│   ├── __init__.py
│   ├── enhanced_retriever.py  # TODO: Enhanced retrieval components
│   ├── knowledge_graph.py     # TODO: KG processing components
│   ├── graph_retriever.py     # TODO: Graph-based retrieval
│   └── hybrid_retriever.py    # TODO: Hybrid retrieval strategies
│
├── pipelines/                  # Specialized processing pipelines
│   ├── __init__.py
│   ├── raw_to_embeddings.py   # Enhanced embeddings pipeline
│   ├── raw_to_knowledge_graph.py # Knowledge graph creation pipeline
│   ├── raw_to_sqlite.py       # SQLite storage pipeline
│   └── hybrid_pipeline.py     # TODO: Combined hybrid pipeline
│
├── evaluation/                 # Specialized evaluation tools
│   ├── __init__.py
│   ├── kg_evaluator.py        # TODO: Knowledge graph evaluation
│   └── hybrid_evaluator.py    # TODO: Hybrid system evaluation
│
├── utils/                      # Specialized utilities
│   ├── __init__.py
│   ├── kg_utils.py            # TODO: Knowledge graph utilities
│   └── db_utils.py            # TODO: Database utilities
│
└── tests/                      # Tests for specialized components
```

## 🚀 Quick Start

### Running Individual Pipelines

```bash
# Run embeddings pipeline
python specialization/main.py --pipeline embeddings --input-dir specialization/data/raw/

# Run knowledge graph pipeline  
python specialization/main.py --pipeline knowledge_graph --input-dir specialization/data/raw/

# Run SQLite pipeline
python specialization/main.py --pipeline sqlite --input-dir specialization/data/raw/

# Run all pipelines
python specialization/main.py --pipeline all --input-dir specialization/data/raw/
```

### Custom Configuration

```bash
# Use custom config file
python specialization/main.py --config path/to/custom_config.py

# Specify custom output directory
python specialization/main.py --pipeline embeddings --output-dir custom/output/path/
```

## 🔧 Configuration

The specialized configuration extends the baseline configuration with:

- **Enhanced Models**: Better embedding models for semantic similarity
- **Knowledge Graph Settings**: Entity extraction and relation confidence thresholds
- **Hybrid Retrieval**: Weights for combining vector and graph-based retrieval
- **Database Settings**: SQLite and vector store configurations
- **Evaluation Metrics**: Extended metrics for specialized evaluation

Key configuration variables:
- `ENHANCED_EMBEDDING_MODEL`: Better embedding model
- `KG_EXTRACTION_MODEL`: Model for entity/relation extraction
- `VECTOR_WEIGHT` / `GRAPH_WEIGHT`: Hybrid retrieval weights
- `SQLITE_DB_NAME`: Database filename

## 📊 Data Management Strategy

### Separation Principle
- **Baseline data** remains in `baseline/data/` - untouched reference
- **Specialized data** goes in `specialization/data/` - isolated environment
- **No cross-contamination** between baseline and specialized datasets

### Data Flow
1. **Raw documents** → `specialization/data/raw/`
2. **Processed outputs** → `specialization/data/processed/`
3. **Embeddings** → `specialization/data/embeddings/`
4. **Knowledge graphs** → `specialization/data/knowledge_graphs/`
5. **Databases** → `specialization/data/db/` and `specialization/data/sqlite/`

## 🛠️ Development Guidelines

### Adding New Components
1. Create component in `components/` folder
2. Add appropriate configuration in `config/config.py`
3. Create unit tests in `tests/`
4. Update this README

### Adding New Pipelines
1. Create pipeline class in `pipelines/` folder
2. Implement `run()` method with consistent interface
3. Add pipeline to `main.py`
4. Add to `pipelines/__init__.py`

### Extending Evaluation
1. Add evaluator in `evaluation/` folder
2. Implement standard evaluation interface
3. Add metrics to configuration
4. Integrate with main pipeline

## 🔗 Integration with Baseline

The specialization module is designed to:
- **Inherit** from baseline where appropriate
- **Override** specific components for enhancement
- **Extend** functionality without breaking baseline
- **Coexist** peacefully with baseline implementation

Import baseline components like this:
```python
from baseline.preprocessor.document_reader import DocumentReader
from baseline.config.config import EMBEDDING_MODEL
```

## 📈 Next Steps

1. **Implement TODO items** in the pipeline files
2. **Add specialized components** for enhanced retrieval
3. **Create evaluation frameworks** for knowledge graphs
4. **Develop hybrid retrieval strategies**
5. **Add comprehensive testing**
6. **Create performance benchmarks**

## 🤝 Contributing

When adding new features:
1. Follow the established folder structure
2. Maintain consistency with baseline interfaces
3. Add appropriate documentation
4. Include tests for new functionality
5. Update configuration as needed
