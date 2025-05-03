# Vector-Databases-with-Chroma-DB-and-FAISS
vector databases to store embeddings generated from textual data using LangChain. The focus will be on two popular vector databases: Chroma DB and Facebook AI Similarity Search (FAISS).
# **Create and Configure a Vector Database to Store Document Embeddings**


## Overview


Imagine you are working in a customer support center that receives a high volume of inquiries and tickets every day. Your task is to create a system that can quickly provide support agents with the most relevant information to resolve customer issues. Traditional methods of searching through FAQs or support documents can be slow and inefficient, leading to delayed responses and dissatisfied customers.

To address this challenge, you will use embedding models to convert support documents and past inquiry responses into numerical vectors that capture their semantic content. These vectors will be stored in a vector database, enabling fast and accurate similarity searches. For example, when a support agent receives a new inquiry about a product issue, the system can instantly retrieve similar past inquiries and their resolutions, helping the agent to provide a quicker and more accurate response.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/veZYoygp9GqZrIw5f6SD0g/vector%20db.png" width="50%" alt="vector db"/>


In this lab, you will learn how to use vector databases to store embeddings generated from textual data using LangChain. The focus will be on two popular vector databases: Chroma DB and Facebook AI Similarity Search (FAISS). You will also learn how to perform similarity searches in these databases based on a query, enabling efficient retrieval of relevant information. By the end of this lab, you will be able to effectively use vector databases to store and query embeddings, enhancing your data analysis and retrieval capabilities.


## __Table of contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
            <li><a href="#Load-text">Load text</a></li>
            <li><a href="#Split-data">Split data</a></li>
            <li><a href="#Embedding model">Embedding model</a></li>
        </ol>
    </li>
    <li>
        <a href="#Vector-store">Vector store</a>
        <ol>
            <li><a href="#Chroma-DB">Chroma DB</a></li>
            <li><a href="#FIASS-DB">FIASS DB</a></li>
            <li><a href="#Managing-vector-store:-adding,-updating,-and-deleting-entries">Managing vector store: adding, updating, and deleting entries</a></li>
        </ol>
    </li>
</ol>

<a href="#Exercises">Exercises</a>
<ol>
    <li><a href="#Exercise-1---Use-another-query-to-conduct-similarity-search.">Exercise 1. Use another query to conduct similarity search.</a></li>
</ol>


## Objectives

After completing this lab, you will be able to:

- Prepare and preprocess documents for embeddings.
- Generate embeddings using watsonx.ai's embedding model.
- Store these embeddings in Chroma DB and FAISS.
- Perform similarity searches to retrieve relevant documents based on new inquiries.


## Setup


For this lab, you will use the following libraries:

* [`ibm-watson-ai`](https://ibm.github.io/watsonx-ai-python-sdk/) for using LLMs from IBM's watsonx.ai.
* [`langchain`, `langchain-ibm`, `langchain-community`](https://www.langchain.com/) for using relevant features from Langchain.
* [`chromadb`](https://www.trychroma.com/) is a open-source vector database used to store embeddings.
* [`faiss-cpu`](https://pypi.org/project/faiss-cpu/) is used to support the using of FAISS vector database.


### Installing required libraries

 __You must run the following cell__ to install them:

**Note:** The version is being pinned here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.

This might take approximately 1-2 minutes. 

As `%%capture` is used to capture the installation, you won't see the output process. After the installation is completed, you will see a number beside the cell.



```python
%%capture
!pip install --user"ibm-watsonx-ai==1.0.4"
!pip install  --user "langchain==0.2.1" 
!pip install  --user "langchain-ibm==0.1.7"
!pip install  --user "langchain-community==0.2.1"
!pip install --user "chromadb==0.4.24"
!pip install  --user "faiss-cpu==1.8.0"
```

The following steps are prerequisite tasks for conducting this project's topic - vector store. These steps include:

- Loading the source document.
- Splitting the document into chunks.
- Building an embedding model.
  
The details of these steps have been introduced in previous lessons.


### Load text


A text file has been prepared as the source document for the downstream vector database task.

Now, let's download and load it using LangChain's `TextLoader`.



```python
!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BYlUHaillwM8EUItaIytHQ/companypolicies.txt"
```

    --2025-04-25 02:57:37--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BYlUHaillwM8EUItaIytHQ/companypolicies.txt
    Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 169.63.118.104
    connected. to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|169.63.118.104|:443... 
    200 OKequest sent, awaiting response... 
    Length: 15660 (15K) [text/plain]
    Saving to: ‘companypolicies.txt’
    
    companypolicies.txt 100%[===================>]  15.29K  --.-KB/s    in 0s      
    
    2025-04-25 02:57:38 (44.5 MB/s) - ‘companypolicies.txt’ saved [15660/15660]
    



```python
# Note: If you see any errors like 'ModuleNotFoundError', please restart the kernal

from langchain_community.document_loaders import TextLoader
```


```python
loader = TextLoader("companypolicies.txt")
data = loader.load()
```

You can have a look at this document.



```python
data
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content="1.\tCode of Conduct\n\nOur Code of Conduct outlines the fundamental principles and ethical standards that guide every member of our organization. We are committed to maintaining a workplace that is built on integrity, respect, and accountability.\nIntegrity: We hold ourselves to the highest ethical standards. This means acting honestly and transparently in all our interactions, whether with colleagues, clients, or the broader community. We respect and protect sensitive information, and we avoid conflicts of interest.\nRespect: We embrace diversity and value each individual's contributions. Discrimination, harassment, or any form of disrespectful behavior is unacceptable. We create an inclusive environment where differences are celebrated and everyone is treated with dignity and courtesy.\nAccountability: We take responsibility for our actions and decisions. We follow all relevant laws and regulations, and we strive to continuously improve our practices. We report any potential violations of this code and support the investigation of such matters.\nSafety: We prioritize the safety of our employees, clients, and the communities we serve. We maintain a culture of safety, including reporting any unsafe conditions or practices.\nEnvironmental Responsibility: We are committed to minimizing our environmental footprint and promoting sustainable practices.\nOur Code of Conduct is not just a set of rules; it is the foundation of our organization's culture. We expect all employees to uphold these principles and serve as role models for others, ensuring we maintain our reputation for ethical conduct, integrity, and social responsibility.\n\n2.\tRecruitment Policy\n\nOur Recruitment Policy reflects our commitment to attracting, selecting, and onboarding the most qualified and diverse candidates to join our organization. We believe that the success of our company relies on the talents, skills, and dedication of our employees.\nEqual Opportunity: We are an equal opportunity employer and do not discriminate on the basis of race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or any other protected status. We actively promote diversity and inclusion.\nTransparency: We maintain transparency in our recruitment processes. All job vacancies are advertised internally and externally when appropriate. Job descriptions and requirements are clear and accurately represent the role.\nSelection Criteria: Our selection process is based on the qualifications, experience, and skills necessary for the position. Interviews and assessments are conducted objectively, and decisions are made without bias.\nData Privacy: We are committed to protecting the privacy of candidates' personal information and adhere to all relevant data protection laws and regulations.\nFeedback: Candidates will receive timely and constructive feedback on their application and interview performance.\nOnboarding: New employees receive comprehensive onboarding to help them integrate into the organization effectively. This includes information on our culture, policies, and expectations.\nEmployee Referrals: We encourage and appreciate employee referrals as they contribute to building a strong and engaged team.\nOur Recruitment Policy is a foundation for creating a diverse, inclusive, and talented workforce. It ensures that we attract and hire the best candidates who align with our company values and contribute to our continued success. We continuously review and update this policy to reflect evolving best practices in recruitment.\n\n3.\tInternet and Email Policy\n\nOur Internet and Email Policy is established to guide the responsible and secure use of these essential tools within our organization. We recognize their significance in daily business operations and the importance of adhering to principles that maintain security, productivity, and legal compliance.\nAcceptable Use: Company-provided internet and email services are primarily meant for job-related tasks. Limited personal use is allowed during non-work hours, provided it doesn't interfere with work responsibilities.\nSecurity: Safeguard your login credentials, avoiding the sharing of passwords. Exercise caution with email attachments and links from unknown sources. Promptly report any unusual online activity or potential security breaches.\nConfidentiality: Reserve email for the transmission of confidential information, trade secrets, and sensitive customer data only when encryption is applied. Exercise discretion when discussing company matters on public forums or social media.\nHarassment and Inappropriate Content: Internet and email usage must not involve harassment, discrimination, or the distribution of offensive or inappropriate content. Show respect and sensitivity to others in all online communications.\nCompliance: Ensure compliance with all relevant laws and regulations regarding internet and email usage, including those related to copyright and data protection.\nMonitoring: The company retains the right to monitor internet and email usage for security and compliance purposes.\nConsequences: Policy violations may lead to disciplinary measures, including potential termination.\nOur Internet and Email Policy aims to promote safe, responsible usage of digital communication tools that align with our values and legal obligations. Each employee is expected to understand and follow this policy. Regular reviews ensure its alignment with evolving technology and security standards.\n\n4.\tMobile Phone Policy\n\nThe Mobile Phone Policy sets forth the standards and expectations governing the appropriate and responsible usage of mobile devices in the organization. The purpose of this policy is to ensure that employees utilize mobile phones in a manner consistent with company values and legal compliance.\nAcceptable Use: Mobile devices are primarily intended for work-related tasks. Limited personal usage is allowed, provided it does not disrupt work obligations.\nSecurity: Safeguard your mobile device and access credentials. Exercise caution when downloading apps or clicking links from unfamiliar sources. Promptly report security concerns or suspicious activities related to your mobile device.\nConfidentiality: Avoid transmitting sensitive company information via unsecured messaging apps or emails. Be discreet when discussing company matters in public spaces.\nCost Management: Keep personal phone usage separate from company accounts and reimburse the company for any personal charges on company-issued phones.\nCompliance: Adhere to all pertinent laws and regulations concerning mobile phone usage, including those related to data protection and privacy.\nLost or Stolen Devices: Immediately report any lost or stolen mobile devices to the IT department or your supervisor.\nConsequences: Non-compliance with this policy may lead to disciplinary actions, including the potential loss of mobile phone privileges.\nThe Mobile Phone Policy is aimed at promoting the responsible and secure use of mobile devices in line with legal and ethical standards. Every employee is expected to comprehend and abide by these guidelines. Regular reviews of the policy ensure its ongoing alignment with evolving technology and security best practices.\n\n5.\tSmoking Policy\n\nPolicy Purpose: The Smoking Policy has been established to provide clear guidance and expectations concerning smoking on company premises. This policy is in place to ensure a safe and healthy environment for all employees, visitors, and the general public.\nDesignated Smoking Areas: Smoking is only permitted in designated smoking areas, as marked by appropriate signage. These areas have been chosen to minimize exposure to secondhand smoke and to maintain the overall cleanliness of the premises.\nSmoking Restrictions: Smoking inside company buildings, offices, meeting rooms, and other enclosed spaces is strictly prohibited. This includes electronic cigarettes and vaping devices.\nCompliance with Applicable Laws: All employees and visitors must adhere to relevant federal, state, and local smoking laws and regulations.\nDisposal of Smoking Materials: Properly dispose of cigarette butts and related materials in designated receptacles. Littering on company premises is prohibited.\nNo Smoking in Company Vehicles: Smoking is not permitted in company vehicles, whether they are owned or leased, to maintain the condition and cleanliness of these vehicles.\nEnforcement and Consequences: All employees and visitors are expected to adhere to this policy. Non-compliance may lead to appropriate disciplinary action, which could include fines, or, in the case of employees, possible termination of employment.\nReview of Policy: This policy will be reviewed periodically to ensure its alignment with evolving legal requirements and best practices for maintaining a healthy and safe workplace.\nWe appreciate your cooperation in maintaining a smoke-free and safe environment for all.\n\n6.\tDrug and Alcohol Policy\n\nPolicy Objective: The Drug and Alcohol Policy is established to establish clear expectations and guidelines for the responsible use of drugs and alcohol within the organization. This policy aims to maintain a safe, healthy, and productive workplace.\nProhibited Substances: The use, possession, distribution, or sale of illegal drugs or unauthorized controlled substances is strictly prohibited on company premises or during work-related activities. This includes the misuse of prescription drugs.\nAlcohol Consumption: The consumption of alcoholic beverages is not allowed during work hours, on company property, or while performing company-related duties. Exception may be made for company-sanctioned events.\nImpairment: Employees are expected to perform their job duties without impairment from drugs or alcohol. The use of substances that could impair job performance or pose a safety risk is prohibited.\nTesting and Searches: The organization reserves the right to conduct drug and alcohol testing as per applicable laws and regulations. Employees may be subject to testing in cases of reasonable suspicion, post-accident, or as part of routine workplace safety measures.\nReporting: Employees should report any concerns related to drug or alcohol misuse by themselves or their colleagues, as well as safety concerns arising from such misuse.\nTreatment and Assistance: Employees with substance abuse issues are encouraged to seek help. The organization is committed to providing support, resources, and information to assist those seeking treatment.\nConsequences: Violation of this policy may result in disciplinary actions, up to and including termination of employment. Legal action may also be pursued when necessary.\nPolicy Review: This policy will undergo periodic review to ensure its continued relevance and compliance with evolving legal requirements and best practices for a safe and productive work environment.\nYour adherence to this policy is appreciated as it helps to maintain a safe and drug-free workplace for all.\n\n7.\tHealth and Safety Policy\n\nOur commitment to health and safety is paramount. We prioritize the well-being of our employees, customers, and the public. We diligently comply with all relevant health and safety laws and regulations. Our objective is to maintain a workplace free from hazards, preventing accidents, injuries, and illnesses. Every individual within our organization is responsible for upholding these standards. We regularly assess and improve our safety measures, provide adequate training, and encourage open communication regarding safety concerns. Through collective dedication, we aim to ensure a safe, healthy, and secure environment for all. Your cooperation is essential in achieving this common goal.\n\n8.\tAnti-discrimination and Harassment Policy\n\nThe Anti-Discrimination and Harassment Policy is a testament to the commitment of this organization in fostering a workplace that is free from discrimination, harassment, and any form of unlawful bias. This policy applies to every individual within the organization, including employees, contractors, visitors, and clients.\nNon-Discrimination: This organization strictly prohibits discrimination based on race, color, religion, gender, national origin, age, disability, sexual orientation, or any other legally protected characteristic in all aspects of employment, including recruitment, hiring, compensation, benefits, promotions, and terminations.\nHarassment: Harassment in any form, whether based on the aforementioned characteristics or any other protected status, is unacceptable. This encompasses unwelcome advances, offensive jokes, slurs, and other verbal or physical conduct that creates a hostile or intimidating work environment.\nReporting: Individuals who experience or witness any form of discrimination or harassment are encouraged to promptly report the incident to their supervisor, manager, or the designated HR representative. The organization is committed to a timely and confidential investigation of such complaints.\nConsequences: Violation of this policy may result in disciplinary action, including termination of employment. The organization is committed to taking appropriate action against any individual found to be in violation of this policy.\nReview and Update: This policy is subject to regular review and update to remain aligned with evolving legal requirements and best practices in preventing discrimination and harassment. This organization considers it a collective responsibility to ensure a workplace free from discrimination and harassment, and it is essential that every individual within the organization plays their part in upholding these principles.\n\n9.\tDiscipline and Termination Policy\n\nThe Discipline and Termination Policy underscores the organization's commitment to maintaining a productive, ethical, and respectful work environment. This policy applies to all personnel, including employees, contractors, and temporary staff.\nPerformance and Conduct Expectations: Employees are expected to meet performance standards and adhere to conduct guidelines. The organization will provide clear expectations, feedback, and opportunities for improvement when performance or conduct issues arise.\nDisciplinary Actions: When necessary, disciplinary actions will be taken, which may include verbal warnings, written warnings, suspension, or other appropriate measures. Disciplinary actions are designed to address issues constructively and maintain performance standards.\nTermination: In situations where an employee's performance or conduct issues persist, the organization may resort to termination. Termination may also occur for reasons such as redundancy, violation of policies, or restructuring.\nTermination Procedure: The organization will follow appropriate procedures, ensuring fairness and adherence to legal requirements during the termination process. Employees may be eligible for notice periods, severance pay, or other benefits as per employment agreements and applicable laws.\nExit Process: The organization will conduct an exit process to ensure a smooth transition for departing employees, including the return of company property, final pay, and cancellation of access and benefits.\nThis policy serves as a framework for handling discipline and termination. The organization recognizes the importance of fairness and consistency in these processes, and decisions will be made after careful consideration. Every employee is expected to understand and adhere to this policy, contributing to a respectful and productive workplace. Regular reviews will ensure its alignment with evolving legal requirements and best practices.\n")]



### Split data


The next step is to split the document using LangChain's text splitter. Here, you will use the `RecursiveCharacterTextSplitter, which is well-suited for this generic text. The following parameters have been set:

- `chunk_size = 100`
- `chunk_overlap = 20`
- `length_function = len`



```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```


```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
```


```python
chunks = text_splitter.split_documents(data)
```

Let's take a look at how many chunks you get.



```python
len(chunks)
```




    215



### Embedding model


The following code demonstrates how to build an embedding model using the `watsonx.ai` package.

For this project, the `ibm/slate-125m-english-rtrvr` embedding model will be used.



```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
```


```python
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)
```

The embedding model is formed into the `watsonx_embedding` object.


## Vector store


In this section, you will be guided on how to use two commonly used vector databases: Chroma DB and FAISS DB. You will also see how to perform a similarity search based on an input query using these databases.


### Chroma DB


#### Build the database


First, you need to import `Chroma` from Langchain vector stores.



```python
from langchain.vectorstores import Chroma
```

Next, you need to create an ID list that will be used to assign each chunk a unique identifier, allowing you to track them later in the vector database. The length of this list should match the length of the chunks.

Note: The IDs should be in string format.



```python
ids = [str(i) for i in range(0, len(chunks))]
```

The next step is to use the embedding model to create embeddings for each chunk and then store them in the Chroma database.

The following code demonstrates how to do this.



```python
vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)
```

Now that you have built the vector store named `vectordb`, you can use the method `.collection.get()` to print some of the chunks indexed by their IDs.

Note: Although the chunks are stored in the database in embedding format, when you retrieve and print them by their IDs, the database will return the chunk text information instead of the embedding vectors.



```python
for i in range(3):
    print(vectordb._collection.get(ids=str(i)))
```

    {'ids': ['0'], 'embeddings': None, 'metadatas': [{'source': 'companypolicies.txt'}], 'documents': ['1.\tCode of Conduct'], 'uris': None, 'data': None}
    {'ids': ['1'], 'embeddings': None, 'metadatas': [{'source': 'companypolicies.txt'}], 'documents': ['Our Code of Conduct outlines the fundamental principles and ethical standards that guide every'], 'uris': None, 'data': None}
    {'ids': ['2'], 'embeddings': None, 'metadatas': [{'source': 'companypolicies.txt'}], 'documents': ['that guide every member of our organization. We are committed to maintaining a workplace that is'], 'uris': None, 'data': None}


You can also use the method `._collection.count()` to see the length of the vector database, which should be the same as the length of chunks.



```python
vectordb._collection.count()
```




    215



#### Similarity search


Similarity search in a vector database involves finding items that are most similar to a given query item based on their vector representations.

In this process, data objects are converted into vectors (which you've already done), and the search algorithm identifies and retrieves those with the closest vector distances to the query, enabling efficient and accurate identification of similar items in large datasets.


LangChain supports similarity search in vector stores using the method `.similarity_search()`.

The following is an example of how to perform a similarity search based on the query "Email policy."

By default, it will return the top four closest vectors to the query.



```python
query = "Email policy"
docs = vectordb.similarity_search(query)
docs
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='internet and email usage, including those related to copyright and data protection.'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='to this policy. Non-compliance may lead to appropriate disciplinary action, which could include'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='This policy serves as a framework for handling discipline and termination. The organization'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Policy Purpose: The Smoking Policy has been established to provide clear guidance and expectations')]



You can specify `k = 1` to just retrieve the top one result.



```python
vectordb.similarity_search(query, k = 1)
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='internet and email usage, including those related to copyright and data protection.')]



### FIASS DB


FIASS is another vector database that is supported by LangChain.

The process of building and using FAISS is similar to Chroma DB.

However, there may be differences in the retrieval results between FAISS and Chroma DB.


#### Build the database


Build the database and store the embeddings to the database here.



```python
from langchain_community.vectorstores import FAISS
```


```python
faissdb = FAISS.from_documents(chunks, watsonx_embedding, ids=ids)
```

Next, print the first three information pieces in the database based on IDs.



```python
for i in range(3):
    print(vectordb._collection.get(ids=str(i)))
```

    {'ids': ['0'], 'embeddings': None, 'metadatas': [{'source': 'companypolicies.txt'}], 'documents': ['1.\tCode of Conduct'], 'uris': None, 'data': None}
    {'ids': ['1'], 'embeddings': None, 'metadatas': [{'source': 'companypolicies.txt'}], 'documents': ['Our Code of Conduct outlines the fundamental principles and ethical standards that guide every'], 'uris': None, 'data': None}
    {'ids': ['2'], 'embeddings': None, 'metadatas': [{'source': 'companypolicies.txt'}], 'documents': ['that guide every member of our organization. We are committed to maintaining a workplace that is'], 'uris': None, 'data': None}


#### Similarity search


Let's do a similarity search again using FIASS DB on the same query.



```python
query = "Email policy"
docs = faissdb.similarity_search(query)
docs
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='internet and email usage, including those related to copyright and data protection.'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='to this policy. Non-compliance may lead to appropriate disciplinary action, which could include'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='This policy serves as a framework for handling discipline and termination. The organization'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Policy Purpose: The Smoking Policy has been established to provide clear guidance and expectations')]



The retrieve results based on the similarity search seem to be the same as with the Chroma DB.

You can try with other queries or documents to see if they follow the same situation.


### Managing vector store: Adding, updating, and deleting entries


There might be situations where new documents come into your RAG application that you want to add to the current vector database, or you might need to delete some existing documents from the database. Additionally, there may be updates to some of the documents in the database that require updating.

The following sections will guide you on how to perform these tasks. You will use the Chroma DB as an example.


#### Add


Imagine you have a new piece of text information that you want to add to the vector database. First, this information should be formatted into a document object.



```python
text = "Instructlab is the best open source tool for fine-tuning a LLM."
```


```python
from langchain_core.documents import Document
```

Form the text into a `Document` object named `new_chunk`.



```python
new_chunk =  Document(
    page_content=text,
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)
```

Then, the new chunk should be put into a list as the vector database only accepts documents in a list.



```python
new_chunks = [new_chunk]
```

Before you add the document to the vector database, since there are 215 chunks with IDs from 0 to 214, if you print ID 215, the document should show no values. Let's validate it.



```python
print(vectordb._collection.get(ids=['215']))
```

    {'ids': [], 'embeddings': None, 'metadatas': [], 'documents': [], 'uris': None, 'data': None}


Next, you can use the method `.add_documents()` to add this `new_chunk`. In this method, you should assign an ID to the document. Since there are already IDs from 0 to 214, you can assign ID 215 to this document. The ID should be in string format and placed in a list.



```python
vectordb.add_documents(
    new_chunks,
    ids=["215"]
)
```




    ['215']



Now you can count the length of the vector database again to see if it has increased by one.



```python
vectordb._collection.count()
```




    216



You can then print this newly added document from the database by its ID.



```python
print(vectordb._collection.get(ids=['215']))
```

    {'ids': ['215'], 'embeddings': None, 'metadatas': [{'page': 1, 'source': 'ibm.com'}], 'documents': ['Instructlab is the best open source tool for fine-tuning a LLM.'], 'uris': None, 'data': None}


#### Update


Imagine you want to update the content of a document that is already stored in the database. The following code demonstrates how to do this.


Still, you need to form the updated text into a `Document` object.



```python
update_chunk =  Document(
    page_content="Instructlab is a perfect open source tool for fine-tuning a LLM.",
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)
```

Then, you can use the method `.update_document()` to update the specific stored information indexing by its ID.



```python
vectordb.update_document(
    '215',
    update_chunk,
)
```


```python
print(vectordb._collection.get(ids=['215']))
```

    {'ids': ['215'], 'embeddings': None, 'metadatas': [{'page': 1, 'source': 'ibm.com'}], 'documents': ['Instructlab is a perfect open source tool for fine-tuning a LLM.'], 'uris': None, 'data': None}


As you can see, the document information has been updated.


#### Delete


If you want to delete documents from the vector database, you can use the method `_collection.delete()` and specify the document ID to delete it.



```python
vectordb._collection.delete(ids=['215'])
```


```python
print(vectordb._collection.get(ids=['215']))
```

    {'ids': [], 'embeddings': None, 'metadatas': [], 'documents': [], 'uris': None, 'data': None}


As you can see, now that document is empty.



```python
query = "Smoking policy"
docs = vectordb.similarity_search(query)
docs
```




    [Document(metadata={'source': 'companypolicies.txt'}, page_content='Smoking Restrictions: Smoking inside company buildings, offices, meeting rooms, and other enclosed'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Designated Smoking Areas: Smoking is only permitted in designated smoking areas, as marked by'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='No Smoking in Company Vehicles: Smoking is not permitted in company vehicles, whether they are'),
     Document(metadata={'source': 'companypolicies.txt'}, page_content='Policy Purpose: The Smoking Policy has been established to provide clear guidance and expectations')]




```python

```
