# TensorFlow - What is it?

I've heard about TensorFlow, seems like it's something to do with Machine Learning. Let's find out!

## Get Setup to Run TensorFlow in a Docker Container
  * [TensorFlow - Docker Installation](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#docker-installation)
  * [Docker on Mac](https://docs.docker.com/mac/step_one/)
  * Before proceeding, does ``docker run hello-world`` work?
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

## Other Resources
* [How to use docker on OSX](https://www.viget.com/articles/how-to-use-docker-on-os-x-the-missing-guide)

 
  
