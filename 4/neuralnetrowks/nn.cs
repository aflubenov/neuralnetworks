using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Linq;

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

    /**
     * for the record: a[colIndex][colVector]
     *                 b[rowIndex][rowVector]
     *
     *    outRes should be: outRes[rowIndex][rowVector]
     */
    static public double[][] matrixMult(double[][] a, double[][] b, ref double[][] outRes){
        Int32 li = a.Length;
        Int32 lj = b.Length;
        double[][] aTmp = outRes;

        var iterations = Enumerable.Range(0, li*lj);
        var pquery = from num in iterations.AsParallel() select num;

        pquery.ForAll((e) => { Int32 row = e/li, col = e%li;  aTmp[row][col] = numpy.dot(a[col], b[row]);} );

        outRes = aTmp;

        return outRes;
    }

    static public double[][] matrixMult(double[][] a, double[][] b){
        Int32 l = b.Length;
        Int32 la = a.Length;
        double[][] aRet = new double[l][];

        for(Int32 i = 0; i < l; i ++)
            aRet[i] = new double[la];

        numpy.matrixMult(a,b, ref aRet);

        return aRet;
    }

    static public double[][] matrixMult(double[] a, double[] b){
        Int32 lb = b.Length;
        Int32 la = a.Length;
        double[][] altera = new double[la][];
        double[][] alterb = new double[lb][];
        double[][] aRet;

        for(Int32 i = 0; i < la; i ++)
            altera[i] = new double[1]{a[i]};

        for(Int32 i = 0; i < lb; i ++)
            alterb[i] = new double[1]{b[i]};

        
        aRet = numpy.matrixMult(a,b);
        
        return aRet;

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

    static public double absSqr(double[] a){
        return dot(a,a);
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
    public double z;
    public double output;
}

public class NeuronsLayer{
    public Int32 nInput = 0;
    public Int32 nNeurons = 0;
    public double[][] inputs = new double[1][]; //1 x number of inputs
    public double[][] weights; //number of neurons x number of inputs
    public double[] biases; //number of neurons 
    public double[][] tmp //number of neurons x 0
    public double[] z; 
    public double[] activation; //number of neurons
    //-------- helper
    public double[] z_prime; //number of neurons
    public double[][] nabla_w; //number of neurons x number of 
    public double[] nabla_b; //number of neurons
 
 
    static public NeuronsLayer(Int32 pnInputs, Int32 pnNeurons){
        this.nInput = pnInputs;
        this.nNeurons = pnNeurons;

        this.inputs[] = new double[pnInputs];
        this.weights = new double[pnNeurons][];
       // this.nabla_w = new double[pnNeurons][];
        for(Int32 i = 0; i < pnInputs; i ++){
            this.weight[i] = numpy.randn1(pnInputs);
           // this.nabla_w[i] = new double[pnInputs];
        }

        this.tmp = new double[pnNeurons][];
        for(Int32 i = 0; i < pnNeurons; i ++)
            this.tmp[i] = new double[1]{0};

        this.biases = numpy.randn1(pnNeurons);
        this.nabla_b = new double[pnNeurons];
        this.z = new double[pnNeurons];
        this.activation = new double[pnNeurons];
        this.z_prime = new double[pnNeurons];
    }
}


public class NeuralNetwork
{
    private Int32 _nLayers;
    private neuronsLayer[] _layers; //first one is the first hidden layer, last one is the output layer 
    private double _learningRate = -0.5;

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
        get { return _layers[_nLayers-1].activation; }
    }

    /***
     * IHlO = #inputs, # of neurons in each layer, #outputs
     * */
    public NeuralNetwork(params Int32[] IHlO)
    {
        Int32 nNeurons, nNeuronsAnt;

        //configuring input information
        _nLayers = IHlO.Length - 1;
        _layers = new neuronsLayer[_nLayers];


        for(Int32 i = 0; i < _nLayers)
            _layers[i] = new NeuronsLayer(IHlO[i],IHlO[i+1]);

    }

    private double[] activateLayer( Int32 i)
    {

        NeuronsLayer current = _layers[i];
        var iterations = Enumerable.Range(0, current.nNeurons);
        var pquery = from num in iterations.AsParallel() select num;

        numpy.matrixMult(current.inputs, current.weights, current.tmp);
        pquery.ForAll((e)=>{ current.z[e]=current.tmp[e][0] + current.biases[e];
                             current.activation[e] = this.Sigmoid(current.z[e]);
                             current.z_prime[e] = this.SigmoidDerivative(current.activation[e]);
                             });
        
    }

    public double[] Feedfordward(double[] inputs)
    {

        _layers[0].inputs[0]=inputs;
        activateLayer(0);

        for (i = 1; i < _nLayers; i++){
            _layers[i].inputs[0]=_layers[i-1];
            activateLayer(i);
        }
        return Activation;
    }

    /// <summary>
    /// adjust weights of a neuron and return the "error feedback" for each weight usefull for the previous neuron layer
    /// </summary>
    /// <param name="i">layer</param>
    /// <param name="error">array</param>
    /// <returns></returns>
    private double[] adjustWeightsAndBias(Int32 i, double[] error )
    {
        Int32 i, l;
        NeuronsLayer current = _layers[i];

        l = current.nNeurons;
        for(i = 0; i < l; i ++){
            current.delta[i] = (error[i])*this.SigmoidDerivative(current.z[i]);
        }


        Int32 i,
            l = neuron.weights.Length;
            Console.WriteLine("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< {0} >>>>>>>>>>>>>>>>>>>>>>>>>>", neuron.z);
        double derivativeError = SigmoidDerivative(neuron.output)*error;
        double[] errorFeedback = neuron.errorFeedback;

        for (i = 0; i < l; i++)
        {
            errorFeedback[i] = derivativeError * neuron.weights[i];
            neuron.weights[i] -= (derivativeError * inputs[i] * _learningRate); 
            
        }

        neuron.bias -= (derivativeError * _learningRate); 

        return errorFeedback;
    }

    public void BackProp(double[] desiredValues)
    {
        
        Int32 i, l;
        NeuronsLayer current = _layers[_nLayers-1]; // we start with the last layer
        double[][] nabla_b = new double[_nLayers][];
        double[][][] nabla_w = new double[_nLayers][][];
        double[] delta = new double[current.nNeurons];

        l = current.nNeurons;
        for(i = 0; i < l; i ++)
            delta[i] = (current.activation[i] - desiredValues[i])*current.z_prime[i];
        
        current.nabla_b = delta;
        current.nabla_w = numpy.matrixMult(current.inputs[0], delta);



        for(i = _nLayers-2; i >=0; i--){
            
        }
        ------------------------------------------------------
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


