using System;
using System.Collections.Generic;

public static class numpy {
	static Random rand = new Random(); //reuse this if you are generating many

	public static double randGauss(){
		double u1 = rand.NextDouble (); //these are uniform(0,1) random doubles
		double u2 = rand.NextDouble ();
		double randStdNormal = Math.Sqrt (-2.0 * Math.Log (u1)) *
			Math.Sin (2.0 * Math.PI * u2); //random normal(0,1)

		double mean = 0; 
		double stdDev = 1;
		return mean + stdDev * randStdNormal;
	}

    public static T[] getArrayPopulated<T>(Int32 size, T value)
    {
        T[] aRet = new T[size];
        for (Int32 i = 0; i < size; i++)
            aRet[i] = value;

        return aRet;
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
		double[][] ret = new double[a[0].Length][];

		for (Int32 i = 0; i < a[0].Length; i++) {
			ret [i] = new double[a.Length];
		}

		for (Int32 i = 0; i < a.Length; i++)
			for (Int32 j = 0; j < a[0].Length; j++)
				ret [j] [i] = a [i] [j];

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
        Int32 l = a.Length;
        double[] ret = new double[l];

        for(Int32 i = 0; i < l; i ++)
            ret[i] = a[i]+b[i];

        return ret;
    }

    static public double[] scalar(double[] a, double s){
        Int32 l = a.Length;
        double[] ret = new double[l];

        for(Int32 i = 0; i < l; i ++)
            ret[i] = a[i]*s;

        return ret;
    }

    static public double[] abs(double[] a){
        Int32 l = a.Length;
        double[] ret = new double[l];

        for(Int32 i = 0; i < l; i ++)
            ret[i] = Math.Abs(a[i]);

        return ret;
    }

    static public double[] sqr(double[] a){
        Int32 l = a.Length;
        double[] ret = new double[l];

        for(Int32 i = 0; i < l; i ++)
            ret[i] = a[i]*a[i];

        return ret;
    }
}




struct struNeuron
{
    public double[] weights;
    public double[] errorFeedback;
    public double bias;
    public double output;
}



public class NeuralNetwork
{
    private Int32 _nInputs;//number of inputs for the ANN
    private double[][] _inputs; //inputs per layer, including the ANN input, _inputs[n] is the input for the n-th hidden layer, the last element of the array is the output

    private Int32 _nLayers;
    private struNeuron[][] _layers; //every layer has and arvitrary number of neurons

    public double[] Activation
    {
        get { return _inputs[_inputs.Length-1]; }
    }

    /***
     * IHlO = #inputs, # of neurons in each layer, #outputs
     * */
    public NeuralNetwork(params Int32[] IHlO)
    {
        Int32 nNeurons, nNeuronsAnt;

        //configuring input information
        _nInputs = IHlO[0];
        _inputs = new double[IHlO.Length][];
        _nLayers = IHlO.Length - 1;

        //configuring hidden layers
        _layers = new struNeuron[_nLayers][];

        nNeuronsAnt = _nInputs;
        for (Int32 i = 0; i < _nLayers; i++)
        {
            nNeurons = IHlO[i+1];
            _layers[i] = new struNeuron[nNeurons];
            for(Int32 j = 0; j < nNeurons; j++)
            {
                _layers[i][j].weights = numpy.randn1(nNeuronsAnt);
                _layers[i][j].errorFeedback = new double[nNeuronsAnt];
                _layers[i][j].bias = numpy.randGauss();
            }
            nNeuronsAnt = nNeurons;
        }
    }

    private double[] activateLayer(double[] inputs, Int32 layerNumber)
    {
        Int32 l = _layers[layerNumber].Length;
        struNeuron neuron;
        double[] aRet = new double[l];

        for (Int32 i = 0; i < l; i++)
        {
            neuron = _layers[layerNumber][i];
            neuron.output = Sigmoid(numpy.dot(neuron.weights, inputs) + neuron.bias);
            _layers[layerNumber][i].output = neuron.output;
            aRet[i] = neuron.output;
        }

        return aRet;
    }

    public double[] Feedfordward(double[] inputs)
    {
        _inputs[0] = inputs;
        Int32 i;

        for (i = 0; i < _nLayers; i++)
            _inputs[i+1] = activateLayer(_inputs[i], i);

        return Activation;
    }

    /// <summary>
    /// adjust weights of a neuron and return the "error feedback" for each weight usefull for the previous neuron layer
    /// </summary>
    /// <param name="error"></param>
    /// <param name="neuron"></param>
    /// <param name="inputs"></param>
    /// <returns></returns>
    private double[] adjustWeightsAndBias(double error, ref struNeuron neuron, double[] inputs)
    {
        Int32 i,
            l = neuron.weights.Length;
        double derivativeError = SigmoidDerivative(neuron.output)*error;
        double[] errorFeedback = neuron.errorFeedback;

        for (i = 0; i < l; i++)
        {
            neuron.weights[i] += (derivativeError * inputs[i]); //remaining learn rate
            errorFeedback[i] = derivativeError * neuron.weights[i];
        }

        neuron.bias += derivativeError; // remaining learn rate

        return errorFeedback;
    }

    public void Feedbackward(double[] desiredValues)
    {
        double[] errors = new double[desiredValues.Length];
        Int32 i, j, k, l = errors.Length;
        struNeuron[] neurons;

        //lets calculate the error from the activation
        for (i = 0; i < l; i++)
            errors[i] = desiredValues[i] - Activation[i];

        //we adjust the weights and bias of the last layer
        i = _nLayers - 1;
        neurons = _layers[i];
        l = neurons.Length;
        for (j = 0; j < l; j++)
            adjustWeightsAndBias(errors[j], ref neurons[j], _inputs[i]);


        //now we ajust the rest of layers
        for (i = _nLayers - 2; i >= 0; i--)
        {
            neurons = _layers[i];
            l = neurons.Length;
            for (j = 0; j < l; j++)
                //every neuron in this layer is the j-th input in every neuron in the next layer
                //So we correct the weight and bias based on the error returned from every neuron in the next layer
                for (k = 0; k < _layers[i + 1].Length; k++)
                    adjustWeightsAndBias(_layers[i + 1][k].errorFeedback[j], ref neurons[j], _inputs[i]);
        }
    }

    private double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    private double SigmoidDerivative(double s)
    {
        return s * (1 - s);
    }

    public void FeedfordwardSet(double[][] inputs)
    {
        Console.Clear();
        for(Int32 i = 0; i < inputs.Length; i++)
        {
            Console.WriteLine("Salida para {0}: {1}", inputs[i][0], Feedfordward(inputs[i])[0]) ;
        }
        System.Threading.Thread.Sleep(10);
    }
    public void pseudoTrain(double[] inputs, double[] desired)
    {
        Feedfordward(inputs);
        Feedbackward(desired);
    }

    public void pseudoTrainSet(double[][] inputs, double[][] desired)
    {
        for (Int32 i = 0; i < inputs.Length; i++)  
            pseudoTrain(inputs[i], desired[i]);
    }
}


public class inicio
{

#region "mierda"

    private static double letterToDouble(String p)
    {
        return Convert.ToUInt16(p.ToCharArray()[0]);
    }
    
    private static double[] stringTodouble(String p)
    {
        ushort[] usA = Array.ConvertAll(p.ToCharArray(), Convert.ToUInt16);
        double[] dRet = new double[usA.Length];

        for (Int32 i = 0; i < dRet.Length; i++)
            dRet[i] = usA[i];

        return dRet;
    }

    public static myType[][] shuffle<myType>(myType[][] array)
    {
        List<myType[]> lNew = new List<myType[]>(array);
        List<myType[]> lRet = new List<myType[]>();
        Random r = new Random();
        myType[] tmp;
        Int32 iTmp;

        while (lNew.Count != 0)
        {
            iTmp = r.Next(0, lNew.Count);
            tmp = lNew[iTmp];
            lRet.Add(tmp);
            lNew.Remove(tmp);
        }

        return lRet.ToArray();
    }

    public static T[][] arrayOfArray<T>(T[] p)
    {
        T[][] aRet = new T[p.Length][];
        for (Int32 i = 0; i < p.Length; i++)
            aRet[i] = new T[1] { p[i] };
        return aRet;
    }


    public static double[] letterToArray(Int32 letterIndex)
    {
        double[] aRet = new double[27];
        aRet[letterIndex] = 1;
        return aRet;
    }

    public static double[] letterToArray(String letter)
    {
        return letterToArray(Convert.ToInt32(letterToDouble(letter)) - 97);
    }



    public static bool pseudoTrainSetUntilNumber(double[][] inputs, double[][] desired, double goodnumber, NeuralNetwork n)
    {
        bool aret = true;
        double[] acumSum = numpy.getArrayPopulated<double>(desired[0].Length, 0);

        //(1 / 2*n)*Sum( abs(activation-desired)^2) where n = number of training cases, and the sum is over every training case.
        n.pseudoTrainSet(inputs, desired);
        for (Int32 i = 0; i < desired.Length; i++)
        {

            acumSum = numpy.add(acumSum, numpy.sqr(numpy.abs(numpy.add(n.Feedfordward(inputs[i]), numpy.scalar(desired[i], -1.0)))));

        }

        acumSum = numpy.scalar(acumSum, 1.0/(2.0*inputs.Length));

        for(Int32 i = 0; i < acumSum.Length; i++)
            aret = aret && (acumSum[i] <= 0.0001);
//Console.WriteLine("aRet es: {0} ", acumSum[0]);
        return aret;
    }

    private static NeuralNetwork recognizeOneLetter()
    {
        NeuralNetwork myNet = new NeuralNetwork( 27, 27, 108, 27, 1 );
        double[][] tryiningData = new double[27][];
        double[][] desired = new double[27][];
        //create 27 letters
        for (Int32 i = 0; i < 27; i++)
        {
            tryiningData[i] = letterToArray(i);
            desired[i] = new double[1] { i < 5 ? 1 : 0 }; //we want letters from a to i, 
        }

        for (; !pseudoTrainSetUntilNumber(tryiningData, desired, 0.99, myNet);)
        {
            //myNet.pseudoTrainSet(tryiningData, desired);
            myNet.FeedfordwardSet(tryiningData);
            //Console.WriteLine("----- ENTRENANDO: {0}", dtmp[0]);
        }

        myNet.FeedfordwardSet(tryiningData);

        String sTmp;
        for (;;)
        {
            Console.WriteLine("Ingrese una letra: ");
            sTmp = Console.ReadLine();
            Console.WriteLine("La salida es: {0}", myNet.Feedfordward(letterToArray(sTmp))[0]);
        }

        return myNet;
    }



    private static NeuralNetwork recognizeDigits()
    {
        NeuralNetwork myNet = new NeuralNetwork( 784, 784, 15, 10 );
        readMNist digits = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        Int32 samples = 20;
        double[][] tryiningData = new double[samples][];
        double[][] desired = new double[samples][];
        Int32 tmp = 0;

        for(Int32 i = 0; i < samples; i ++){
            digits.GiveNextValue(out tryiningData[i], ref tmp);
            desired[i]=numpy.getArrayPopulated<double>(10,0);
            desired[i][tmp-1]=1;
        }

        for (; !pseudoTrainSetUntilNumber(tryiningData, desired, 0.99, myNet);)
        {
            //myNet.pseudoTrainSet(tryiningData, desired);
            myNet.FeedfordwardSet(tryiningData);
            //Console.WriteLine("----- ENTRENANDO: {0}", dtmp[0]);
        }

        myNet.FeedfordwardSet(tryiningData);

        String sTmp;
        for (;;)
        {
            Console.WriteLine("Ingrese una letra: ");
            sTmp = Console.ReadLine();
            Console.WriteLine("La salida es: {0}", myNet.Feedfordward(letterToArray(sTmp))[0]);
        }

        return myNet;
    }



    private static NeuralNetwork recognizeAnd()
    {
        NeuralNetwork myNet = new NeuralNetwork( 2, 5, 15, 15, 1 );


        double[][] desired = new double[4][];
        double[][] dTryiningData = new double[4][];
        double[] dtmp;

        desired[0] = new double[1] { 1 };
        desired[1] = new double[1] { 1 };
        desired[2] = new double[1] { 0 };
        desired[3] = new double[1] { 0 };

        dTryiningData[0] = new double[2] { 0, 0 };
        dTryiningData[1] = new double[2] { 0, 1 };
        dTryiningData[2] = new double[2] { 1, 0 };
        dTryiningData[3] = new double[2] { 1, 1 };

        for (; myNet.Feedfordward(dTryiningData[1])[0] <= 0.99;) // || myNet.feedFordward(dTryiningData[0])[0] >= 0.7;)
        {
            myNet.pseudoTrainSet(dTryiningData, desired);
            myNet.FeedfordwardSet(dTryiningData);
            //Console.WriteLine("----- ENTRENANDO: {0}", dtmp[0]);
        }

        
        string sTmp;

        for (;;)
        {
            dtmp = new double[2];
            sTmp = Console.ReadLine();
            dtmp[0] = double.Parse(sTmp);

            sTmp = Console.ReadLine();
            dtmp[1] = double.Parse(sTmp);

            dtmp = myNet.Feedfordward(dtmp);
            Console.Write("Salida personal -------{0} ", dtmp[0]);
        }

        return myNet;
    }
#endregion

    public static void Main()
    {

       // readMNist a = new readMNist("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        
        //recognizeOneLetter();
        recognizeDigits();

    }
}