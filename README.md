# AI Tools Assignment - Theoretical Answers & Ethics

## Part 1: Theoretical Understanding (40%)

### 1. Short Answer Questions

**Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?**

**Key Differences:**
- **Execution Model**: TensorFlow uses static computation graphs (define-then-run), while PyTorch uses dynamic graphs (define-by-run)
- **Learning Curve**: PyTorch is more intuitive for beginners due to its Pythonic nature, while TensorFlow has a steeper learning curve
- **Deployment**: TensorFlow has better production deployment tools (TensorFlow Serving, TensorFlow Lite), while PyTorch is catching up with TorchServe
- **Community**: Both have strong communities, but PyTorch is preferred in research, TensorFlow in industry

**When to Choose:**
- **TensorFlow**: Production deployment, mobile apps, large-scale distributed training, established MLOps pipelines
- **PyTorch**: Research projects, prototyping, dynamic models (RNNs with variable sequences), educational purposes

**Q2: Describe two use cases for Jupyter Notebooks in AI development.**

1. **Exploratory Data Analysis (EDA)**: Interactive environment for data visualization, statistical analysis, and hypothesis testing before building models
2. **Model Prototyping & Experimentation**: Iterative development of ML models with immediate feedback, allowing data scientists to test different approaches quickly

**Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?**

spaCy provides:
- **Linguistic Intelligence**: Part-of-speech tagging, dependency parsing, named entity recognition
- **Pre-trained Models**: Language-specific models with deep understanding of grammar and semantics
- **Efficiency**: Optimized C extensions for fast processing of large text corpora
- **Pipeline Architecture**: Modular components that can be customized and extended

### 2. Comparative Analysis: Scikit-learn vs TensorFlow

| Aspect | Scikit-learn | TensorFlow |
|--------|--------------|------------|
| **Target Applications** | Classical ML (regression, classification, clustering) | Deep Learning, Neural Networks, Large-scale ML |
| **Ease of Use** | Very beginner-friendly, simple API | Steeper learning curve, more complex |
| **Community Support** | Excellent documentation, stable API | Large community, frequent updates, extensive ecosystem |

## Part 3: Ethics & Optimization (10%)

### 1. Ethical Considerations

**Potential Biases in Models:**

**MNIST Model Biases:**
- **Dataset Bias**: MNIST contains mainly Western handwriting styles, may not generalize to other cultural writing patterns
- **Quality Bias**: Clean, centered digits may not reflect real-world messy handwriting
- **Demographic Bias**: Training data may not represent handwriting from all age groups or abilities

**Amazon Reviews Model Biases:**
- **Selection Bias**: Reviews may not represent all customer demographics
- **Language Bias**: English-only processing excludes non-English speakers
- **Temporal Bias**: Older reviews may not reflect current product quality

**Mitigation Strategies:**

1. **TensorFlow Fairness Indicators**: 
   - Evaluate model performance across different demographic groups
   - Use metrics like equalized odds and demographic parity
   - Implement bias detection in the ML pipeline

2. **spaCy Rule-based Systems**:
   - Create inclusive entity recognition rules
   - Implement multi-language processing
   - Use balanced training data for custom models

### 2. Troubleshooting Challenge

**Common TensorFlow Bugs & Fixes:**

```python
# Common Bug 1: Dimension Mismatch
# Wrong:
model.add(Dense(10, input_shape=(784)))  # Missing comma
# Fixed:
model.add(Dense(10, input_shape=(784,)))  # Correct tuple

# Common Bug 2: Incorrect Loss Function
# Wrong:
model.compile(loss='categorical_crossentropy', ...)  # For sparse labels
# Fixed:
model.compile(loss='sparse_categorical_crossentropy', ...)  # Correct for integer labels

# Common Bug 3: Data Type Mismatch
# Wrong:
x_train = x_train / 255  # Integer division
# Fixed:
x_train = x_train.astype('float32') / 255.0  # Proper normalization
```

## Implementation Notes

### Required Dependencies
```bash
pip install tensorflow scikit-learn spacy matplotlib pandas numpy
python -m spacy download en_core_web_sm
pip install streamlit  # For bonus task
```

### Key Results Expected:
- **Iris Classification**: >95% accuracy (typically achieves ~100%)
- **MNIST CNN**: >95% accuracy (our model should achieve ~98%+)
- **spaCy NLP**: Successful entity extraction and basic sentiment analysis

### Ethical AI Development Best Practices:
1. **Diverse Training Data**: Ensure representation across demographics
2. **Bias Testing**: Regular evaluation for fairness across groups
3. **Transparency**: Document model limitations and biases
4. **Continuous Monitoring**: Track model performance over time
5. **Inclusive Design**: Consider accessibility and cultural differences

## Conclusion

This assignment demonstrates proficiency in:
- Classical ML with Scikit-learn
- Deep Learning with TensorFlow
- NLP processing with spaCy
- Ethical considerations in AI development
- Practical deployment strategies

The minimal code approach ensures clarity while maintaining functionality across all required tasks.
