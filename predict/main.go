package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Recognition runs mnist recognition.
func Recognition(tensor *tensorflow.Tensor) (string, error) {
	var probability float64
	// tf.saved_model.builder in Pythonで構築したモデルを呼び出す
	model, err := tensorflow.LoadSavedModel("./../train/conv_mnist_pb", []string{"serve"}, nil)
	if err != nil {
		return "", err
	}
	defer model.Session.Close()

	result, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation("conv2d_1_input").Output(0): tensor,
		},
		[]tensorflow.Output{
			model.Graph.Operation("dense_2/Softmax").Output(0),
		},
		nil,
	)

	if err != nil {
		return "", err
	}
	labels := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
	probabilities := result[0].Value().([][]float32)[0]
	max := 0
	fmt.Println(probabilities)
	for i, v := range probabilities {
		if float64(v) > probability {
			probability = float64(probabilities[i])
			max = i
		}
	}
	return labels[max], nil
}

// ConvertImg2Tensor converts image to tensor.
func ConvertImg2Tensor(imageBuffer *bytes.Buffer) (*tensorflow.Tensor, error) {
	format := "jpg"
	tensor, err := tensorflow.NewTensor(imageBuffer.String())
	if err != nil {
		return nil, err
	}
	graph, input, output, err := makeTransFormImageGraph(format)
	if err != nil {
		return nil, err
	}
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{input: tensor},
		[]tensorflow.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func makeTransFormImageGraph(format string) (graph *tensorflow.Graph, input, output tensorflow.Output, err error) {
	const (
		Height, Width = 28, 28
	)
	s := op.NewScope()
	input = op.Placeholder(s, tensorflow.String)
	var decode tensorflow.Output
	decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(1)) //0,1だけなので1

	// decodeした画像のtensorにbatch sizeを加える
	decodeWithBatch := op.ExpandDims(
		s,
		op.Cast(s, decode, tensorflow.Float),
		op.Const(s.SubScope("make_batch"), int32(0)),
	)
	// imageを28x28にリサイズ
	output = op.ResizeBilinear(
		s,
		decodeWithBatch,
		op.Const(s.SubScope("size"), []int32{Height, Width}),
	)
	graph, err = s.Finalize()
	return graph, input, output, err
}

func main() {
	fmt.Println(tensorflow.Version())
	var imageBuffer bytes.Buffer
	io.Copy(&imageBuffer, os.Stdin)
	tensor, err := ConvertImg2Tensor(&imageBuffer)
	if err != nil {
		log.Fatal(err)
	}
	class, err := Recognition(tensor)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(class)
}
