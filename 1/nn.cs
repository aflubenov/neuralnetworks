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

	static public double[] add(double[] a, double[] b){
		Int32 i = 0,
			l = a.Length;

		double[] ret = new double[a.Length];

		for (i = 0; i < l; i ++)
			ret [i] = a [i] + b [i];

		return ret;
	}


}



public class NNetwork {

	private Int32 num_layers;
	private Int32[] sizes;
	private double[][][] biases;
	private double[][][] weights;

	private double sigmoid(double z){
		return 1.0 / (1.0 + Math.Exp (-z));
	}

	private double[][] sigmoid(double[][] z){
		double[][] ret = new double[z.Length][];

		for (Int32 i = 0; i < z.Length; i ++)
			ret [i] = new double[z [i].Length];

		for (Int32 i = 0; i < ret.Length; i ++)
			for (Int32 j = 0; j < ret[i].Length; j++)
				ret [i] [j] = sigmoid (z [i] [j]);

		return ret;
	}

	/*private void feedfordward(double[] a){
		Int32 l =num_layers-1;

		for(Int32 i = 0; i < l; i ++){
			a = 
		}

	}*/

	public NNetwork(Int32[] sizes){
		Int32 i = 0;

		num_layers = sizes.Length;
		this.sizes = sizes;

		biases = new double[num_layers - 1][][];

		for (i = 1; i < num_layers; i++) {
			biases [i-1] = numpy.randn (this.sizes [i],1);
		}

		weights = new double[num_layers - 1][][];
		for (i = 1; i < num_layers; i++) {
			weights [i-1] = numpy.randn (sizes [i], sizes [i - 1]);
		}

	}
}


public class inicio {
    public static void Main() {
		NNetwork mired = new NNetwork (new Int32[6] { 4, 5, 4,3,2,1 });


    }
}