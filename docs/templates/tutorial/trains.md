# Trains Integration

Allegro Trains is a full system open source ML / DL experiment manager and ML-Ops solution.
It enables data scientists and data engineers to effortlessly track, manage, compare and collaborate on their experiments as well as easily manage their training workloads on remote machines.

**Trains** is a suite of open source Python packages and plugins, including:

* [**Trains**](https://github.com/allegroai/trains) Python Client package - Integrate **Trains** into your AutoKeras tasks with just two lines of code, and get all of **Trains** robust features. 
* [**Trains Server**](https://github.com/allegroai/trains-server) - The **Trains** backend infrastructure and web UI. Use the public [**Trains** demo server](https://demoapp.trains.allegro.ai), or deploy your own.
* [**Trains Agent**](https://github.com/allegroai/trains-agent) -  The **Trains** DevOps component for experiment execution, resource control, and autoML..
* Additional integrations - Integrate **Trains** with [PyCharm](https://github.com/allegroai/trains-pycharm-plugin) and [Jupyter Notebook](https://github.com/allegroai/trains-jupyter-plugin). 

<img src="https://allegro.ai/docs/img/trains/gif/webapp_screenshots.gif">

## Setting up Trains

To integrate **Trains** into your AutoKeras project, do the following:

1. Install the **Trains** Python Client package.

        pip install trains

1. Add the short **Trains** initialization code to your task.

        from trains import Task
        
        task = Task.init(project_name="autokeras", task_name="autokeras imdb example with scalars")

1. Run your task. The console output will include the URL of the task's **RESULTS** page.
    
        TRAINS Task: overwriting (reusing) task id=60763e04c0ba45ea9fe3cfe79f3f06a3
        TRAINS results page: https://demoapp.trains.allegro.ai/projects/21643e0f1c4a4c99953302fc88a1a84c/experiments/60763e04c0ba45ea9fe3cfe79f3f06a3/output/log</code></pre>

See an example script [here](https://github.com/allegroai/trains/blob/master/examples/autokeras/autokeras_imdb_example.py).

## Tracking your AutoKeras tasks
### Visualizing Task Results

**Trains** automatically logs comprehensive information about your AutoKeras task: code source control, execution environment, hyperparameters and more.  
It also automatically records any scalars, histograms and images reported to Tensorboard/Matplotlib or Seaborn.

For example, making use of Tensorboard in your task will make all recorded information available in **Trains** as well:

```python
from tensorflow import keras

tensorboard_callback_train = keras.callbacks.TensorBoard(log_dir='log')
tensorboard_callback_test = keras.callbacks.TensorBoard(log_dir='log')
clf.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback_train])
clf.fit(x_test, y_test, epochs=2, callbacks=[tensorboard_callback_test])
```

When your task runs, you can follow its results, including any collected metrics through the **Trains** web UI.

View your task results in the **Trains** web UI, by clicking on it in the **EXPERIMENTS** table.  
Find the **EXPERIMENT** table under the specified project listed in the **HOME** or **PROJECTS** page:  

<img src="https://allegro-datasets.s3.amazonaws.com/erez/Selection_028.png" style="border: 1px solid black; border-radius:3px">


Detailed description **Trains** Web UI experiment information can be obtained [here](https://allegro.ai/docs/webapp/webapp_exp_details/).  
Additional information on **Trains** logging capabilities can be obtained in the [relevant **Trains** Documentation](https://allegro.ai/docs/concepts_arch/concepts_arch/#logging)

### Task Models

**Trains** automatically tracks models produced by your AutoKeras tasks.

To upload models, specify the `output_uri` parameter when calling `Task.init` to provide the upload destination:

        task = Task.init(project_name="autokeras", 
            task_name="autokeras imdb example with scalars",
            output_uri="http://localhost:8081/")

View models information in the experiment details panel, **ARTIFACTS** tab:  

<img id="myImg_01" class="modalImg" src="https://allegro-datasets.s3.amazonaws.com/erez/Selection_029.png" style="border: 1px solid black; border-radius:3px">

### Tracking Model Performance

Use the **Trains** web UI to easily create experiment leaderboards and quickly identify best performing models.  
Customize your board adding any valuable metric or hyperparameter.

<img id="myImg_03" class="modalImg" src="https://allegro-datasets.s3.amazonaws.com/erez/Selection_031.png" style="border: 1px solid black; border-radius:3px">  

Additional information on customizing **Trains** experiment and model tables can be obtained in the [relevant **Trains** Documentation](https://allegro.ai/docs/webapp/webapp_exp_table/#customize-the-experiments-table)

### Model Development Insights

Use the **Trains** web UI to view side-by-side comparison of experiments: Easily locate the differences and impact of experiment configuration parameters, metrics, scalars etc.

Compare multiple experiments, by selecting two or more experiments in the **EXPERIMENTS** table, and clicking **COMPARE**.

The following image shows how two experiments compare in their epoch_accuracy and epoch_loss behaviour:

<img id="myImg_02" class="modalImg" src="https://allegro-datasets.s3.amazonaws.com/erez/Selection_030.png" style="border: 1px solid black; border-radius:3px">