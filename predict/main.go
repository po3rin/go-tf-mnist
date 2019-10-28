package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"

	"github.com/po3rin/resize"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Recognition runs recognition with TensorFlow.
func Recognition(tensor *tensorflow.Tensor) (string, error) {
	fmt.Println(tensor.DataType())
	fmt.Println(tensor.Shape())
	fmt.Println(tensor.Value())
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
		return "", fmt.Errorf("faield to run session: %w", err)
	}
	probabilities := result[0].Value().([][]float32)[0]

	labels := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
	var max int
	var probability float64
	for i, v := range probabilities {
		if float64(v) > probability {
			probability = float64(probabilities[i])
			max = i
		}
	}
	return labels[max], nil
}

// img2TensorForMNIST prepares tensor (reshape 28 * 28 & convert gray scale)
func img2TensorForMNIST(img image.Image) (*tensorflow.Tensor, error) {
	b := img.Bounds()
	if b.Max.X != 28 || b.Max.Y != 28 {
		img = resize.Resize(img, 28, 28)
		b = img.Bounds()
	}

	els := make([][][]float32, 28)
	for i := range els {
		els[i] = make([][]float32, 28)
	}

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, _, _, _ := color.GrayModel.Convert(img.At(x, y)).RGBA()
			els[y][x] = []float32{float32(int(r / 255))}
		}
	}

	imageData := [][][][]float32{els}
	return tensorflow.NewTensor(imageData)
}

func main() {
	img, err := jpeg.Decode(os.Stdin)
	if err != nil {
		log.Fatalf("failed to decode jpeg: %v", err)
	}
	tensor, err := img2TensorForMNIST(img)
	if err != nil {
		log.Fatalf("failed to convert image to tensor: %v", err)
	}
	result, err := Recognition(tensor)
	if err != nil {
		log.Fatalf("failed to recognit: %v", err)
	}

	fmt.Println(result)
}
