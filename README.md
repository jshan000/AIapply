# Smart Legal Form Builder: AI-Enhanced Document Creation


## 📢 Group Members

| Name          | Organization                                              | Email                        |
|---------------|-----------------------------------------------------------|------------------------------|
| Seoyeon Kim   | Department of Tourism (관광학부), Hanyang University       | soen0814@hanyang.ac.kr        |
| Jiwon Han     | Department of Business Administration (경영학부), Hanyang University | jiwon8623@hanyang.ac.kr       |
| Taeyeon Kim   | Department of Information System (정보시스템학과), Hanyang University | tangboll5075@gmail.com        |
| Suhyeon Yu     | Department of Information System (정보시스템학과), Hanyang University | sh265826@naver.com            |
| Soohan Jeong  | Department of Information System (정보시스템학과), Hanyang University | soohan9909@gmail.com          |



## 📖 Topic - Smart Legal Form Builder: AI-Enhanced Document Creation

### I. Introduction
The Smart Legal Form Builder is designed for individuals who struggle with legal document preparation due to a lack of legal expertise or access to affordable professional assistance. Many people face challenges in understanding and drafting legal documents because they lack formal legal education or experience, leaving them unsure about the structure, required elements, and proper legal terminology. This often leads to errors or delays in processing important legal documents.

According to the 2019 국민법의식통계 (Korean Legal Awareness Statistics), approximately 76.3% of respondents found legal terminology difficult to understand, and 78.4% struggled with comprehending legal sentences. In the same study, 61.6% of respondents evaluated that legal services in society are quantitatively insufficient. Additionally, hiring a lawyer or legal professional can be expensive and time-consuming. Simple document preparation often involves significant financial investment, and individuals with limited budgets may feel discouraged from pursuing legal remedies or formal agreements. A 2020 report from the Judicial Policy Research Institute highlights that legal service costs pose a major burden to the public, prompting discussions on introducing legal service insurance systems.

The AI-powered legal form builder addresses these issues by offering a user-friendly interface that guides individuals through the process of creating formal documents, such as complaints. The tool is specifically tailored to assist in the preparation of legal documents for cases involving fraud in secondhand transaction fraud, online abuse, sexual harassment, assault and injury. It simplifies the creation of high-quality legal documents for non-experts by providing step-by-step guidance via a chatbot interface that gathers the necessary information.

This solution not only enables users to create legal documents without professional assistance but also improves accessibility to legal solutions by offering a cost-effective alternative to traditional legal services. By making legal support more accessible, the tool helps underrepresented communities and small businesses navigate legal challenges efficiently and affordably.


---


### II. Datasets
The dataset used to train the **Complaint Generation AI Model** consists of three **CSV files**, each containing different sections of a **legal complaint**. These datasets are provided in text format and include the essential information required to create a complaint, such as complainant details, defendant details, and incident descriptions.

#### CSV files (sample)

- [Harassment Dataset](./harassment_dataset.csv)
- [Online Incident Reports](./online_incident_reports.csv)
- [중고거래사기500 Dataset](./중고거래사기500_dataset.csv)


#### Dataset Structure:

1. Complainant Information:
   - Name, Address, Phone Number, National ID, Occupation
2. Defendant Information:
   - Name, Address, Phone Number, National ID, Occupation
3. Incident Description:
   - Incident Date, Location, Incident Details, Outcome
4. Complaint Intent:
   - Desired legal outcome
5. Attached Documents:
   - Medical Certificate, Witness Statements

Each of these fields is provided in text format, and they are structured in a way that allows the AI model to generate a **complaint** following a standardized format.

#### Dataset Generation Method

The dataset was generated using **GPT-2**, which was employed to produce various text formats and variations required for complaint generation. The model directly generated data for complaint templates that include complainant information, defendant information, incident descriptions, and complaint intent, which were used as training data.

Reasons for creating the dataset with GPT-2:

1. **Lack of comprehensive legal document datasets**: Most available datasets for complaints are either limited in scope or only cover specific types of incidents. This limitation makes it difficult to train a model that can generalize across various legal scenarios. Additionally, legal terminology and the variability of incidents are complex, making it essential to create diverse templates to ensure the model learns to handle multiple types of complaints.

2. **Leveraging GPT's data augmentation capabilities**: By using GPT, we were able to automatically generate a wide variety of complaint documents for different legal situations. This method enabled us to expand the training dataset without manual effort and allowed for more generalized learning. The diverse set of generated complaints gave the model the flexibility to handle a wide range of real-world legal scenarios, making the system more powerful and adaptable.


---


### III. **Methodology**

#### **1. Model and Environment Setup**

In this research, we chose the **GPT-2** model for complaint generation. GPT-2 is a **natural language generation model** that specializes in **text generation** and is highly effective in creating coherent and contextually relevant sentences. By leveraging GPT-2, we aim to automate the process of creating legal documents such as complaints, ensuring the generated text adheres to legal standards.

The environment setup for fine-tuning the model involves installing the following essential libraries:
- Python 3.8+
- Hugging Face Transformers library
- PyTorch or TensorFlow (based on preference)
- Additional utilities: datasets, accelerate, wandb (optional)

```bash
pip install transformers datasets accelerate wandb
```

Once the environment is set up, we load the **GPT-2 model** and the **tokenizer** for text preprocessing.

#### **2. Dataset Preparation**

For this project, we use a text dataset formatted in plain Text for training the model. The dataset is composed of text entries that represent **legal document sections** such as **complainant details**, **defendant information**, **incident description**, and **complaint intent**. This data is used to train the model to generate legal documents based on the provided templates.

**Dataset Structure**:
1. **Complainant Information**:
   - Name, Address, Phone Number, National ID, Occupation
2. **Defendant Information**:
   - Name, Address, Phone Number, National ID, Occupation
3. **Incident Description**:
   - Incident Date, Location, Incident Details, Outcome
4. **Complaint Intent**:
   - Desired legal outcome
5. **Attached Documents**:
   - Medical Certificate, Witness Statements

The dataset is divided into four categories based on specific legal issues that are commonly encountered:

Secondhand Transaction Fraud: Legal cases involving fraudulent transactions in secondhand goods, such as selling goods that do not meet the agreed-upon standards or failing to deliver products after receiving payment.
Online Abuse: Legal cases related to harassment, defamation, or abuse occurring through digital platforms such as social media or online messaging services.
Sexual Harassment: Cases involving unwanted sexual advances or behaviors, often within the workplace or public spaces.
Assault and Injury: Legal matters involving physical harm or threats of harm, such as assault, battery, or injury due to negligence or criminal activity.
Each of these categories is used to generate a specific legal complaint, and the model learns to handle the nuances of different types of legal cases.

Here is an example of how to load a dataset:

```python
from datasets import load_dataset

# Load the dataset (example for public datasets)
dataset = load_dataset("path_to_dataset")

# For local text dataset
from datasets import Dataset
data = {"text": ["sentence 1", "sentence 2", "..."]}
dataset = Dataset.from_dict(data)
```

#### **3. Data Preprocessing and Tokenization**

Before training the model, we need to tokenize the text data. Tokenization breaks down the sentences into tokens (smaller chunks of text) that the model can understand. The text is preprocessed and tokenized using the GPT-2 tokenizer.

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def preprocess_data(examples):
    inputs = ["Generate complaint: " + text for text in examples["command"]]
    
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    labels = tokenizer(examples["query"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    
    return {"input_ids": model_inputs["input_ids"], "attention_mask": model_inputs["attention_mask"], "labels": labels["input_ids"]}
```

This preprocessing step ensures that the dataset is in a format suitable for training the model.

#### **4. Model Fine-Tuning**

Fine-tuning involves adapting the pre-trained GPT-2 model to our specific task of complaint generation. The model is trained on the dataset that includes complaint templates and various legal document elements. This process adjusts the model’s weights to enable it to generate text that follows legal structures and templates.

We create a blank complaint template to guide the training process, which includes 4 primary sections: complainant details, defendant information, incident description, and complaint intent. The model learns to fill in these sections accurately from the dataset.

The dataset is divided into four categories based on specific legal cases:

1. Secondhand Transaction Fraud
2. Online Abuse
3. Sexual Harassment
4. Assault and Injury

The fine-tuning process involves setting up the training arguments and using the Trainer class to initiate the training.

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",          # Directory for saving results
    num_train_epochs=3,              # Number of epochs
    per_device_train_batch_size=8,   # Batch size
    learning_rate=5e-5,              # Learning rate
)

trainer = Trainer(
    model=model,                     # Model to train
    args=training_args,              # Training configuration
    train_dataset=train_dataset,     # Training dataset
    tokenizer=tokenizer,             # Tokenizer
)

trainer.train()
```

Through fine-tuning, the model learns to generate legal text based on complaint templates.

#### **5. Model Evaluation and Validation**

Once the model is trained, it is evaluated using a validation dataset. Evaluation metrics include accuracy, precision, recall, and F1-score to assess how well the model performs in generating legally accurate complaints.

Additionally, to ensure legal accuracy, the generated complaints are reviewed by legal experts to verify that they meet the required legal standards. This validation step ensures that the model not only generates text but also complies with legal requirements.

#### **6. Hyperparameter Optimization**

The training process includes the tuning of hyperparameters to optimize the model’s performance. Hyperparameters such as learning rate, batch size, and number of epochs are adjusted based on experimental results. This optimization helps to achieve better results in generating legally compliant text.


By fine-tuning the GPT-2 model with complaint templates, this methodology enables the automatic generation of legally compliant complaints**. The process includes training the model on high-quality datasets, fine-tuning it to adhere to legal document standards, and evaluating its performance through multiple metrics. The system provides a significant advancement in automating the generation of legal documents and can be applied to various legal tasks, making legal services more accessible to individuals and small businesses.


---


### IV. Evaluation & Analysis

The Smart Legal Form Builder is currently under training, and performance evaluation and analysis will be conducted after the model is fully trained and validated. However, based on the current development progress, the following evaluation and analysis directions have been set.

#### 1. Evaluation Metrics

The key metrics for evaluating the model's performance are accuracy, precision, recall, and F1-score. These metrics will play a crucial role in evaluating how accurately the model generates text that meets the legal standards for complaint generation.

- Accuracy: Measures how well the generated complaints match the templates and legal requirements.
- Precision: Evaluates how accurately the model generates legally relevant information in the complaint text.
- Recall: Ensures that all essential elements required for the complaint (e.g., complainant details, incident description) are included in the generated text.
- F1-score: Provides a balance between precision and recall, giving an overall performance metric.


#### 2. Initial Results and Analysis

Based on the initial model results, the basic details such as complainant and defendant information are generated accurately. However, more complex sections like the incident description and legal outcomes need further refinement. There is a potential for inaccuracies in describing the incident details or legal outcomes, which will require adjustments in future iterations.

Additionally, data imbalance remains a concern, especially for categories like sexual harassment and assault, where there is limited training data. This imbalance may result in lower model accuracy for these specific cases.

#### 3. Improvement Strategy

The following improvement strategies will be employed to enhance the model’s performance:

- Data Augmentation: Increase the training data for various incident categories to improve the model’s generalization capabilities.
- Hyperparameter Tuning: Optimize the model's learning rate, batch size, and epoch count to enhance its performance.
- Fine-tuning: Further fine-tune the model to specialize in different incident types, ensuring that the legal requirements for each category are met effectively.

---

### V. Related Work

There have been various studies and projects that have explored AI-based legal document generation and natural language processing (NLP) techniques in the legal domain.

#### 1. Legal Text Generation Using NLP

In recent years, many studies have focused on natural language processing (NLP) for legal text generation, particularly using models like BERT, GPT-2, and T5. These models are used to automatically generate legal documents, such as contracts or complaints, with a focus on generating legally accurate and contextually correct texts. For example, Sutskever et al. (2014) developed Sequence-to-Sequence models for text generation, which have been applied to legal documents in recent years to improve text consistency and legal accuracy.

#### 2. Complaint Generation Using GPT Models

While there has been less research specifically on complaint generation, the GPT model has been applied to legal document generation. Studies using GPT-3 for text generation have shown promising results in generating complaints, contracts, and judgments. However, these models still face challenges in maintaining legal accuracy and ensuring that the generated text fully meets legal standards, which often requires expert legal review.

#### 3. Challenges in Legal Document Automation

A significant challenge in automating legal documents is ensuring legal accuracy and compliance with legal standards. Several projects, such as Deep Legal and LexPredict, have attempted to address these challenges by using machine learning and NLP technologies to improve the generation of legal documents, particularly by ensuring legal terms and document structure are properly understood and followed.

#### 4. Legal Document Automation Platforms

Several popular legal document automation platforms offer tools for creating, managing, and storing legal documents. These platforms use pre-built templates and guided questionnaires to help users create customized legal documents, similar to the AI-driven approach used in this project. Some notable platforms include:

- Rocket Lawyer: A versatile platform that provides users with tools to create, manage, and store legal documents. It offers a variety of pre-built templates for common legal needs, including contracts, leases, and wills. The platform streamlines document creation with guided questionnaires, allowing users to customize templates with case-specific details. Rocket Lawyer also connects users with licensed attorneys to provide on-demand legal advice and document reviews, offering a balance of automation and human assistance.
  
- LegalZoom: One of the most well-known platforms for creating personalized legal documents online. It offers an extensive library of templates and provides users with a guided process to input relevant details and create custom documents for personal and business use. LegalZoom’s services extend beyond document creation, including trademark registration, business formation assistance, and access to expert legal advice. While not specialized in complaints handling, LegalZoom is a great solution for individuals and businesses looking for efficient and accessible legal tools.
  
- LawDepot: A platform that enables users to create legal documents quickly and efficiently. It offers a wide range of customizable templates, including contracts, agreements, and estate planning documents. By answering a series of guided questions, users can create documents tailored to their specific situation. LawDepot is especially useful for individuals seeking a cost-effective solution for everyday legal matters without the need for professional assistance.

These platforms, like the Smart Legal Form Builder, aim to simplify the legal document creation process, making legal services more accessible and affordable. However, while these platforms focus on user-driven document creation, the Smart Legal Form Builder aims to automatically generate legal complaints, offering a more hands-off solution that can enhance efficiency and accuracy for users with less legal expertise.

---

### VI. Conclusion: Discussion

In this research, the Smart Legal Form Builder was developed using the GPT-2 model to automate the generation of legal complaints. The model focuses on filling in the required information such as complainant details, defendant information, incident description, and complaint intent, based on structured templates.

#### Key Findings:
- The model has generated complaints with basic information (e.g., complainant details, defendant details) with high accuracy.
- However, the incident description and legal outcomes sections still need improvement in terms of accuracy and legal compliance.

#### Future Directions:
To improve the model's performance, we plan to augment the dataset to address data imbalance, especially for categories like sexual harassment and assault. Additionally, fine-tuning the model for different legal cases and optimizing hyperparameters will help improve its overall performance. Legal accuracy will be a key focus of further improvements, with an emphasis on incorporating expert legal reviews.

The Smart Legal Form Builder presents a significant step toward automating legal document generation. By improving its accuracy and generalization, this model has the potential to make legal services more accessible and efficient for individuals, especially in cost-sensitive legal contexts.


## 📗 Related Documents
[Smart Legal Form Builder PPT](./Smart%20Legal%20Form%20Builder%20PPT.pdf)  

[Smart Legal Form Builder Report.pdf](https://github.com/jshan000/AIapply/blob/main/Smart%20Legal%20Form%20Builder%20Report.pdf)

## 🔗 Code Link

**Frontend**

https://github.com/jshan000/VIBE-client

**Backend**

https://github.com/jshan000/VIBE-server
