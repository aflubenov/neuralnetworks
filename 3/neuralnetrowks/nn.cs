using System;
using System.Collections.Generic;
using System.IO;

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
    private double _learningRate = 3;

    static public NeuralNetwork getFromFile(string fileName){
        BinaryReader oReader = new BinaryReader(File.Open(fileName,FileMode.Open));
        Int32 inputs;
        Int32 layersNumber; 
        Int32[] creationParams;
        NeuralNetwork oRet;

        inputs = oReader.ReadInt32(); //number of inputs
        oReader.ReadInt32(); //number or layers (first dimention) for _inputs[n]
        layersNumber = oReader.ReadInt32(); //number of layers for _nLayers;

        creationParams = new Int32[layersNumber+1];
        creationParams[0] = inputs;
        for(Int32 i = 0; i < layersNumber; i ++)
            creationParams[i+1] = oReader.ReadInt32();

        oRet = new NeuralNetwork(creationParams);
        oRet.loadFromFile(oReader);
        oReader.Close();

        return oRet;
    }

    public void loadFromFile(BinaryReader oReader){
        Int32 lNeurons;
        Int32 lWeights;

        //the file should be already opened and the file pointer should be in the begining of
        //the layers iteration to load neurons

        for(Int32 i = 0; i < _nLayers; i ++){
            lNeurons = oReader.ReadInt32(); //number of neurons for this layer
            for(Int32 j = 0; j < lNeurons; j++){
                _layers[i][j].bias = oReader.ReadDouble(); //bias
                oReader.ReadInt32(); //feedback array size
                lWeights = oReader.ReadInt32(); //number of weights for this neuron
                for(Int32 k = 0; k < lWeights; k++)
                    _layers[i][j].weights[k]=oReader.ReadDouble();
            }
        }
    }

    public void saveToFile(string fileName){
        BinaryWriter oWritter = new BinaryWriter(File.Open(fileName,FileMode.Create));
        //inputs number
        //inputs layer lengths  (first dimmention)
        //layers number
        //the number of neuros for each layer
        //for every layer
          //number of neurons
            //neuron bias
            //error feedback array size
            //number of weights
                //weights 

        oWritter.Write(_nInputs);
        oWritter.Write(_inputs.Length);
        oWritter.Write(_nLayers);

        //we write the number of neurons of every layer, this is important for the loading from file.
        for(Int32 i = 0; i < _nLayers; i ++)
            oWritter.Write(_layers[i].Length);

        //now, for every layer, we write number of neurons and then the neurons
        for(Int32 i = 0; i < _nLayers; i ++){
            oWritter.Write(_layers[i].Length); //number of neurons for this layer
            //for every neuron, we write its values
            for(Int32 j = 0; j < _layers[i].Length; j ++){
                oWritter.Write(_layers[i][j].bias); //we write the bias
                oWritter.Write(_layers[i][j].errorFeedback.Length); //the size of error feedback array
                //now the number of weights and the weights
                oWritter.Write(_layers[i][j].weights.Length);
                for(Int32 k = 0; k < _layers[i][j].weights.Length; k ++)
                    oWritter.Write(_layers[i][j].weights[k]);
            }
        }

        oWritter.Close();

    }


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
            neuron.weights[i] += (derivativeError * inputs[i] * _learningRate); //remaining learn rate
            errorFeedback[i] = derivativeError * neuron.weights[i];
        }

        neuron.bias += (derivativeError * _learningRate); // remaining learn rate

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

    public void FeedfordwardSet(double[][] inputs, double[][] desired)
    {
        
        System.Text.StringBuilder texto = new System.Text.StringBuilder("");
        double[] salida;
        System.Text.StringBuilder[] esperado = new System.Text.StringBuilder[desired.Length];

        for(Int32 i = 0; i < desired.Length; i ++){
            esperado[i] = new System.Text.StringBuilder("    Esperado: ");
            for(Int32 j = 0; j < desired[i].Length; j++)
                esperado[i].Append(String.Format(" {0:N0}", desired[i][j]));
        }


        for(Int32 i = 0; i < inputs.Length; i++)
        {
            salida = Feedfordward(inputs[i]);
            //texto.Clear();
            for(Int32 j = 0; j < salida.Length; j ++)
                texto.Append(String.Format(" {0:N9}", salida[j]));
            //Console.WriteLine(texto) ;
            texto.Append(esperado[i].ToString()+"\n");
        }
        Console.SetCursorPosition(0,0);
        
        Console.WriteLine(texto.ToString());
//        System.Threading.Thread.Sleep(10);
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


