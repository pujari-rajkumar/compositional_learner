<h3>Overview</h3>

This is the code repository for the paper, Understanding Politics via Contextualized Discourse Processing: <tt> https://rajkumar-pujari.com/understanding_politics.html </tt>


<h3>Data Download</h3>
Download the required data from here: <tt> https://drive.google.com/file/d/1Y-BTF34tnS8FhE0dGViY9MYN7nNrb50N/view?usp=sharing </tt>

Extract the zipped archive and place <tt>data/</tt> folder inside the <tt>compositional_learner/</tt> folder and alongside <tt>compositional_reader/</tt>, <tt>data_processing/</tt> etc,. Make sure that there isn't a nested folder named <tt>data/</tt> inside the <tt>data/</tt> folder after extraction. Folders inside <tt>data/</tt> should be <tt>composite_learner_data/</tt>, <tt>evaluation_data/</tt> and so on.


<h3>Experiments</h3>
    Follow the steps sequentially to reproduce results from the paper. This repository consists of scripts related to data collection, data processing, learnings tasks and evaluation tasks. If you are only interested in reproducing evaluation results, you don't need to train the entire model again. Trained model parameters are provided in the downloadable data link. Skip forward to evaluation tasks section below.
    If you are interested in training compositional reader model on learning tasks as described in the paper, you don't need to run data collection scripts. Data required to train the model, as described in the paper, is provided in the downloadable data link. Skip forward to the data pre-processing section below.


<h3>Data Collection</h3>
To collect data using collection scripts:
TO DO


<h3>Data Processing</h3>
To pre-process data for learning tasks:

	1. Download staford-corenlp from here: https://stanfordnlp.github.io/corenlp-docs-dev/download.html

	2. Extract and place the contents in the main directory in a folder named staford-corenlp/

	3. Run the following commands in this order:

                                    python data_processing/text_processing.py
                                    python data_processing/document_bert_embedding_computation.py
				    python data_processing/wikification.py
                                    python data_processing/query_dict_generation.py

P.S: Don't run them in parallel as the later steps require the output of the earlier steps. Data pre-processing takes a long time (~5 hours on a machine with 256 cores and 356 GB RAM)


<h3>Learning Tasks</h3>
To train the compositional reader model on learning tasks, you first need to run the data pre-processing scripts. They will create <tt>query_dicts-\*.pkl</tt> files in <tt>compostional_learner/data/composite_learner_data/data_examples/</tt> folder. Then, you need to run:

                                    python learning_tasks/authorship_prediction_model.py
                                    python learning_tasks/mentioned_entity_prediction_model.py

Upon running the two commands sequentially, you should be able to see the training progress in <tt>compositional_learner/data/composite_learner_data/training_logs/</tt> folder. Trained_models will be saved in <tt>compositional_learner/data/composite_learner_data/saved_parameters/</tt> folder. You may use the saved paramters to initialize the parameters for further tasks. Alternatively, you may use the parameters provided in the data download link above for inference.


<h3>Evaluation Tasks</h3>
To run evaluation tasks:

1. To generate visualizations for 'politicians on all issues' (Fig. 5 and Appendix Fig.s 4-13) and 'comaprison of politicians stances on issues' (Fig. 4), you may run:

                                    python ./evaluation_tasks/visualizations.py

	It runs on CPU and will take 2-5 secs to finish

2. To generate results of 'NRA Grades Paraphrase Task' (Tab. 4), 'NRA Grade Prediction Task' (Tab 5., Fig. 4 and Appendix Fig. 2), LCV Score Prediction Task (Appendix Tab. 1, Fig.s 1 & 2), you may run:
 
                                    python ./evaluation_tasks/grade_prediction.py


	This also runs on CPU and may take up to 30 minutes to finish. It is trains and evaluates a GradePredictor feed-forward neural network 320 times (5 random seeds * 8 training data sizes * 4 models * 2 tasks). We didn't provide the option of GPU because it doesn't take too much time to finish.


3. Both the scripts will generate plots as <tt>.png</tt>  images in the <tt>compositional_learner/data/evaluation_data/</tt> folder. <tt>grade_prediction.py</tt> generates training output in <tt>compositional_learner/data/evaluation_data/grade_pred_log.txt</tt>.

4. We provided representations generated by all models for all entities for all issues in the <tt>compositional_learner/data/evaluation_data/</tt> folder. They are named <tt>entity_issue_<model_name>.pkl</tt>.
