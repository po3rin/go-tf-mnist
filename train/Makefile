clean:
	rm -r conv_mnist_pb	&& rm mnist.h5 mnist.json model.h5

pb:
	freeze_graph --input_saved_model_dir=./conv_mnist_pb/ --output_graph=conv_mnist.pb --output_node_names=dense_2/Softmax --clear_devices && cp conv_mnist.pb ./conv_mnist_pb

train:
	docker run -it -v $(pwd):/home python:3.7 bash

install:
	pip install -r requirements.txt
