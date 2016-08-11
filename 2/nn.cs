using System;
using System.Collections.Generic;

public static class numpy {
	static Random rand = new Random(); //reuse this if you are generating many

	static double randGauss(){
		double u1 = rand.NextDouble (); //these are uniform(0,1) random doubles
		double u2 = rand.NextDouble ();
		double randStdNormal = Math.Sqrt (-2.0 * Math.Log (u1)) *
			Math.Sin (2.0 * Math.PI * u2); //random normal(0,1)

		double mean = 0; 
		double stdDev = 1;
		return mean + stdDev * randStdNormal;
	}

	static public double[] randn1(int n){
		double[] ret = new double[n];

		for (Int32 i = 0; i < n; i ++) {
			ret [i] = randGauss ();
		}

		return ret;
	}

	static public double[][] randn(int n, int d){
		double[][] ret = new double[n][];

		for (Int32 i = 0; i < n; i ++) {
			ret [i] = randn1 (d);
		}

		return ret;
	}

	static public double dot(double[] a, double[] b){
		Int32 l = a.Length;
		double ret = 0.0;
		for (Int32 i = 0; i < l; i ++) {
			ret += (a [i] * b [i]);
		}

		return ret;
	}

	static public double[][] traspose(double[][] a){
		double[][] ret = new double[a.Length][];

		for (Int32 i = 0; i < a.Length; i++) {
			ret [i] = new double[a [i].Length];
		}

		for (Int32 i = 0; i < a.Length; i++)
			for (Int32 j = 0; j < a[0].Length; j++)
				ret [i] [j] = a [j] [i];

		return ret;
	}

	static public double[][] dot(double [][] a, double [][] b){
		double[][] ret = new double[a.Length][];
		double[][] bt = traspose (b);

		for (Int32 i = 0; i < a.Length; i ++)
			ret [i] = new double[b[0].Length];

		for (Int32 i = 0; i < a.Length; i ++)
			for (Int32 j = 0; j < b[0].Length; j ++)
				ret [i] [j] = dot (a [i], bt [j]);

		return ret;

	}


}



public class Neuron {

	private double _bias;
	private double[] _weights;
	private double[] _inputs;
	private double _output = double.MinValue;
	private double _error = 0;

	public double[] weights {
		get { return _weights;}
		set { _weights = value;}
	}

	double bias {
		get { return _bias;}
		set { _bias = value;}
	}

	public double[] inputs {
		get { return _inputs; }
		set { _inputs = value;}
	}
	public double sigmoid(double z) {
		return 1.0 / (1.0 + Math.Exp (-z));
	}

	public double derivative(){
		double s = this.output();
		return s * (1 - s);
	}

	public Neuron(double pBias){
		this.bias = pBias;
	}

	public void adjustWeights(double error) {
		Int32 i,
			l = this._inputs.Length;

		this._error = error;

		for(i = 0; i < l; i ++)
			this._weights[i] += (error * this.derivative()*this._inputs[i]); //remaining learn rate

		this._bias += error * this.derivative(); // remaining learn rate

		this._output = double.MinValue;
	}

	public double errorFeedback(Int32 inputIndex){
		return this._error * this.derivative() * this._weights[inputIndex];
	}


	public double output(double[] i, double[] w) {
		this.inputs = i;
		this.weights = w;
		this._output = double.MinValue;
		return output ();
	}

	public double output(double[] i){
		this.inputs = i;
		this._output = double.MinValue;
		return output();
	}

	public double output(){
		if(this._output != double.MinValue)
			return this._output;

		this._output = sigmoid(numpy.dot(this.inputs, this.weights) + this.bias);
		return this._output;
	}


}


public class NNetwork {

	private Int32 num_layers;
	private Int32[] sizes;

	private Neuron[][] layers;

	
	private void setBiases(Int32 layerIndex, double[] biases){

		layers[layerIndex] = new Neuron[biases.Length];
		 
		for (Int32 i = 0; i < biases.Length; i ++)
			layers [layerIndex] [i] = new Neuron (biases [i]);
	}

	private void setWeights(Int32 layerIndex, Int32 neuronIndex, double[] weights){
		layers [layerIndex] [neuronIndex].weights = weights;
	}

	private void setWeights(Int32 layerIndex, double[][] weights){
		for (Int32 i = 0; i < weights.Length; i ++)
			setWeights (layerIndex, i, weights [i]);
	}

	public double[] feedFordward(double[] inputs){
		Int32 l = num_layers - 1;
		Int32 i=0;
		double[] layerOuput = inputs;

		for (i = 0; i < l; i ++) {
			layerOuput = this.feedFordward (layerOuput, i);
		}

		return layerOuput;
	}

	public double[] feedFordward(double[] inputs, Int32 layer){
	
		Neuron[] neurons = layers[layer];
		Int32 l = neurons.Length;
		Int32 i;
		double[] ret = new double[l];

		//every neuron has just one output and many inputs
		//the set of all the outputs are the inputs for the next layer of neurons
		//so we collect the outputs of every neuron and return it as an array
		for (i = 0; i < l; i ++) {
			ret [i] = neurons [i].output (inputs);
		}

		return ret;
	}

	public void feedbackward(double desired)


	public void train(double[][] inputs, double[] desired){
		Int32 l = inputs.Length;
		Int32 i;
		Int32 j, lNeurons;
		double[] outputs = new double[l];
		double[] errors = new double[l];
		Neuron[] layer;

		/*
		1.- Initialise the network with small random weights.
		2.- Present an input pattern to the input layer of the network.
		3.- Feed the input pattern forward through the network to calculate its activation value.
		4.- Take the difference between desired output and the activation value to calculate the network’s activation error.
		5.- Adjust the weights feeding the output neuron to reduce its activation error for this input pattern.
		6.- Propagate an error value back to each hidden neuron that is proportional to their contribution of the network’s activation error.
		7.- Adjust the weights feeding each hidden neuron to reduce their contribution of error for this input pattern.
		8.- Repeat steps 2 to 7 for each input pattern in the input collection.
		9.- Repeat step 8 until the network is suitably trained.
		*/

		//1 initialize network (done!)
		//2 inputs
		//3 feed network and calculate values 
		for(i = 0; i < l; i ++) {
			outputs[i] = this.feedFordward(inputs[i]);
			// 4 calculate difference between desired against ouput  
			errors[i] = desired[i] - outputs[i];
		}

		
		layer = layers[layers.Length-1];
		lNeurons = layer.Length;

		for(i = 0; i < l; i ++)
			for(j = 0; j < lNeurons; j ++)
				//5 adjust weight of output (last) neuron
				layer[j].adjustWeights(errors[i]);  //TODO - from here

		//6 propagate error value back since de last but one layer
		for(i = num_layers-2; i >=0; i --){
			layers[i].adjustWeights()
		}
		
		

	}


	public NNetwork(Int32[] sizes){
		Int32 i = 0;
		double[] biasesTmp;

		num_layers = sizes.Length;
		this.sizes = sizes;

		//we deprecate the first layer because it is the input layer
		layers = new Neuron[num_layers -1][];

		//for each layer, we set as many biases as the size of the layer 
		//using numbers from the gaussian distribution
		for (i = 1; i < num_layers; i++) {
			biasesTmp = numpy.randn1 (this.sizes [i]);
			setBiases (i - 1, biasesTmp);
		}

		//we set the weights.
		//
		//numpy.randn(x,y) returns an array double[x][y] where
		//"x" = number of neurons in that layer, and "y" = number
		//of weights to set, that is the same as the number of inputs 
		//that neuron "xi" will have, in this case, every neuron "xi" has as
		//many inputs as neurons in the previous layer.
		for (i = 1; i < num_layers; i++) {
			setWeights(i-1, numpy.randn (sizes [i], sizes [i - 1]));
		}

	}


}


public class inicio {
    public static void Main() {
		NNetwork mired = new NNetwork (new Int32[3] { 5, 5, 1 });

		String entrada = System.Console.ReadLine();

		ushort[] usEntrada = Array.ConvertAll(entrada.ToCharArray(), Convert.ToUInt16);

		double[] dEntrada = new double[usEntrada.Length];
		for(Int32 i = 0; i < dEntrada.Length;i++ )
			dEntrada[i] = usEntrada[i];

		double[] a = mired.feedFordward(dEntrada);
		Console.WriteLine(a[0]);
    }
}