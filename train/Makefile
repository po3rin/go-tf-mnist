clean:
	rm -r conv_mnist_pb	&& rm mnist.h5 mnist.json

pb:
	freeze_graph --input_saved_model_dir=./conv_mnist_pb/ --output_graph=conv_mnist.pb --output_node_names=dense_2/Softmax --clear_devices && copy conv_mnist.pb ./conv_mnist_pb

train:
	docker run -it tensorflow/tensorflow:1.13.1 --volume ./:/home/

install:
	pip install -r requirements.txt
