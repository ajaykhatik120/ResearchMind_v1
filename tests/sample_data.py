from langchain_core.documents import Document

documents = [

    # ==========================================
    # CATEGORY 1: DEEP LEARNING
    # ==========================================
    
    Document(
    page_content=(
        "ResNet-50 is a 50-layer deep convolutional neural network "
        "that uses residual connections to train very deep networks."

        ),
    metadata={
        "category": "deep_learning",
        "topic": "resnet"

        }
    ),


    Document(
        page_content="Vanishing gradients occur when gradients become extremely small during backpropagation through deep neural networks.",
        metadata={"category": "deep_learning", "topic": "vanishing_gradients"}
    ),

    Document(
        page_content="Very small gradients prevent earlier neural network layers from learning effectively.",
        metadata={"category": "deep_learning", "topic": "vanishing_gradients"}
    ),

    Document(
        page_content="ReLU activation helps reduce the vanishing gradient problem compared with sigmoid activation.",
        metadata={"category": "deep_learning", "topic": "activation_functions"}
    ),

    Document(
        page_content="Backpropagation calculates gradients using the chain rule to update neural network weights.",
        metadata={"category": "deep_learning", "topic": "backpropagation"}
    ),

    Document(
        page_content="CNNs use convolutional filters to extract spatial features from images.",
        metadata={"category": "deep_learning", "topic": "cnn"}
    ),

    # ==========================================
    # CATEGORY 2: MACHINE LEARNING
    # ==========================================

    Document(
        page_content="Decision trees make predictions by recursively splitting data using feature-based conditions.",
        metadata={"category": "machine_learning", "topic": "decision_trees"}
    ),

    Document(
        page_content="Random forests combine multiple decision trees using bagging to improve prediction stability.",
        metadata={"category": "machine_learning", "topic": "random_forest"}
    ),

    Document(
        page_content="Gradient boosting builds sequential models where each new model attempts to correct previous errors.",
        metadata={"category": "machine_learning", "topic": "gradient_boosting"}
    ),

    Document(
        page_content="Support vector machines find a decision boundary that maximizes the margin between classes.",
        metadata={"category": "machine_learning", "topic": "svm"}
    ),

    # ==========================================
    # CATEGORY 3: STATISTICS
    # ==========================================

    Document(
        page_content="The mean is calculated by dividing the sum of observations by the number of observations.",
        metadata={"category": "statistics", "topic": "mean"}
    ),

    Document(
        page_content="Standard deviation measures how spread out observations are around their mean.",
        metadata={"category": "statistics", "topic": "standard_deviation"}
    ),

    Document(
        page_content="Bayes theorem calculates conditional probability using prior probability and likelihood.",
        metadata={"category": "statistics", "topic": "bayes_theorem"}
    ),

    Document(
        page_content="Hypothesis testing uses sample evidence to evaluate a claim about a population.",
        metadata={"category": "statistics", "topic": "hypothesis_testing"}
    ),

    # ==========================================
    # CATEGORY 4: FINANCE
    # ==========================================

    Document(
        page_content="Portfolio diversification reduces investment risk by spreading capital across different assets.",
        metadata={"category": "finance", "topic": "portfolio_diversification"}
    ),

    Document(
        page_content="The Sharpe ratio measures risk-adjusted investment returns relative to volatility.",
        metadata={"category": "finance", "topic": "sharpe_ratio"}
    ),

    Document(
        page_content="The CAPM model estimates expected stock returns using systematic risk and market returns.",
        metadata={"category": "finance", "topic": "capm"}
    ),

    Document(
        page_content="Markowitz portfolio optimization balances expected returns against portfolio variance.",
        metadata={"category": "finance", "topic": "portfolio_optimization"}
    ),

    # ==========================================
    # CATEGORY 5: GENERAL KNOWLEDGE
    # ==========================================

    Document(
        page_content="The Indian education system includes schools, colleges and universities.",
        metadata={"category": "general", "topic": "education"}
    ),

    Document(
        page_content="The stock market provides a platform for buying and selling financial securities.",
        metadata={"category": "general", "topic": "stock_market"}
    ),

    Document(
        page_content="Climate change refers to long-term changes in global temperature and weather patterns.",
        metadata={"category": "general", "topic": "climate_change"}
    )
]

