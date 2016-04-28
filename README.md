# TensorFlow - What is it?

I've heard about TensorFlow, seems like it's something to do with Machine Learning. Let's find out!

## Get Setup to Run TensorFlow in a Docker Container
  * [TensorFlow - Docker Installation](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#docker-installation)
  * [Docker on Mac](https://docs.docker.com/mac/step_one/)
  * Before proceeding, does ``docker run hello-world`` work?
    * If not, make sure you tried the ``Docker Quickstart Terminal`` and are running ``docker run hello-world`` in that terminal window
    * Check out ``/Applications/Docker/Docker\ Quickstart\ Terminal.app/Contents/Resources/Scripts/start.sh`` if you're interested in more details about how this works
  * Get the docker IP address
    * ``$ docker-machine ip default``
      * -> ``192.168.99.100`` Whatever this IP address is copy it to the clipboard 
  * Run the tensorflow image, with port forwarding:
    * ``$ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow``
      * -> open a web browser to http://{docker ip address}:8888/
      * So for my example that is [http://192.168.99.100:8888/](http://192.168.99.100:8888/)


## Where are we?
At this point it looks like we have [jupyter](http://jupyter.org/) running in a docker container
* What is jupyter?
  * Something about [Notebooks](http://ipython.org/notebook.html)
  * IPython Notebooks are now [Jupyter Notebooks](http://jupyter.org/)
  * [Notebooks are a supported format on GitHub](http://blog.jupyter.org/2015/05/07/rendering-notebooks-on-github/)

## Where next?
* Looks like the notebooks are our key to learning more:
  * On your docker container open the ``1_hello_tensorflow.ipynb`` Notebook, for me:
    * [http://192.168.99.100:8888/notebooks/1_hello_tensorflow.ipynb](http://192.168.99.100:8888/notebooks/1_hello_tensorflow.ipynb)
  * I'm now going to read it
  * Look's like you can highlight a 'cell' with Python code in it and run the code by clicking the ``>|`` button in the document toolbar!
  * The result of the code is displayed beneath the 'cell'

## Other Resources
* [How to use docker on OSX](https://www.viget.com/articles/how-to-use-docker-on-os-x-the-missing-guide)

 
  
