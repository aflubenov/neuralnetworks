using System;

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

	public double[] weights {
		get { return _weights;}
		set { _weights = value;}
	}

	double bias {
		get { return _bias;}
		set { _bias = value;}
	}

	public double sigmoid(double z){
		return 1.0 / (1.0 + Math.Exp (-z));
	}

	public double sigmoidDerivative(double x){
		double s = sigmoid (x);
		return s * (1 - s);
	}

	public Neuron(double pBias){
		this.bias = pBias;
	}


	public double output(double[] i, double[] w) {
		this.weights = w;
		return output (i);
	}

	public double output(double[] i){
		return sigmoid(numpy.dot(i, this.weights) + this.bias);
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
		NNetwork mired = new NNetwork (new Int32[3] { 784, 30, 10 });

		mired.ToString ();
    }
}