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
    static public double[][] matrixMult_T1(double[][] a, double[][] b, ref double[][] outRes){
        Int32 li = a.Length;
        Int32 lj = b.Length;
        double[][] aTmp = outRes;

        var iterations = Enumerable.Range(0, li*lj);
        var pquery = from num in iterations.AsParallel() select num;

        pquery.ForAll((e) => { Int32 row = e/li, col = e%li;  aTmp[row][col] = numpy.dot(a[col], b[row]);} );

        outRes = aTmp;

        return outRes;
    }

    static public double[][] matrixMult_T1(double[][] a, double[][] b){
        Int32 l = b.Length;
        Int32 la = a.Length;
        double[][] aRet = new double[l][];

        for(Int32 i = 0; i < l; i ++)
            aRet[i] = new double[la];

        numpy.matrixMult_T1(a,b, ref aRet);

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

        
        aRet = numpy.matrixMult_T1(altera,alterb);
        
        return aRet;

    }

    static public double[][] matrixMult(double[][] a, double[][] b){
        double[][] aT = numpy.traspose(a);
        double[][] aRet;
        aRet = numpy.matrixMult_T1(aT, b);
        return aRet;
    }

    static public double[][] matrixMult(double[][] a, double[] b){
        double[][] alterB = new double[1][];
        double[][] aRet;

        alterB[0] = b;

        aRet = matrixMult(a,alterB);
        return aRet;
    }

    static public double[] hadamart(double[] a, double[] b){
        Int32 i, l = a.Length;
        double[] aRet = new double[l];

        for(i = 0; i < l; i ++)
            aRet[i] = a[i]*b[i];

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
    public double[][] tmp; //number of neurons x 0
    public double[] z; 
    public double[] activation; //number of neurons
    //-------- helper
    public double[] z_prime; //number of neurons
    public double[][] nabla_w; //number of neurons x number of
    public double[][] nabla_w_tmp;

    public double[] nabla_b; //number of neurons
    public double[] nabla_b_tmp;
 
    public NeuronsLayer(Int32 pnInputs, Int32 pnNeurons){
        this.nInput = pnInputs;
        this.nNeurons = pnNeurons;

        this.inputs[0] = new double[pnInputs];
        this.weights = new double[pnNeurons][];
       // this.nabla_w = new double[pnNeurons][];
        for(Int32 i = 0; i < pnInputs; i ++){
            this.weights[i] = numpy.randn1(pnInputs);
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

    public void cleanNablaTmp(){
        nabla_b_tmp = new double[nNeurons];
        for(Int32 i = 0; i < nNeurons; i ++)
            nabla_b_tmp[i] = 0;

        nabla_w_tmp = new double[nNeurons][];
        for(Int32 i = 0; i < nNeurons; i ++){
            nabla_w_tmp[i] = new double[nInput];
            for(Int32 j = 0; j < nInput; j++)
                nabla_w_tmp[i][j]=0;
        }


    }
}


public class NeuralNetwork
{
    private Int32 _nLayers;
    private Int32 _nInputs;
    private NeuronsLayer[] _layers; //first one is the first hidden layer, last one is the output layer 
    private double _learningRate = -0.5;

    static public NeuralNetwork getFromFile(string fileName){
        BinaryReader oReader = new BinaryReader(File.Open(fileName,FileMode.Open));
        Int32 inputs;
        Int32 layersNumber; 
        Int32[] creationParams;
        NeuralNetwork oRet;

        inputs = oReader.ReadInt32(); //number of inputs
        oReader.ReadInt32(); //nothing ... just backward compatibility
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
                _layers[i].biases[j] = oReader.ReadDouble(); //bias
                oReader.ReadInt32(); //nothing, just backward compatibility
                lWeights = oReader.ReadInt32(); //number of weights for this neuron
                for(Int32 k = 0; k < lWeights; k++)
                    _layers[i].weights[j][k]=oReader.ReadDouble();
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
        oWritter.Write(1); //do nothing, just backward compatibility ///// oWritter.Write(_inputs.Length);
        oWritter.Write(_nLayers);

        //we write the number of neurons of every layer, this is important for the loading from file.
        for(Int32 i = 0; i < _nLayers; i ++)
            oWritter.Write(_layers[i].nNeurons);

        //now, for every layer, we write number of neurons and then the neurons
        for(Int32 i = 0; i < _nLayers; i ++){
            oWritter.Write(_layers[i].nNeurons); //number of neurons for this layer
            //for every neuron, we write its values
            for(Int32 j = 0; j < _layers[i].nNeurons; j ++){
                oWritter.Write(_layers[i].biases[j]); //we write the bias
                oWritter.Write(1); //do nothing, just backward compatibility ///////////oWritter.Write(_layers[i][j].errorFeedback.Length); //the size of error feedback array
                //now the number of weights and the weights
                oWritter.Write(_layers[i].weights[j].Length);
                for(Int32 k = 0; k < _layers[i].weights[j].Length; k ++)
                    oWritter.Write(_layers[i].weights[j][k]);
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
//        Int32 nNeurons, nNeuronsAnt;

        //configuring input information
        _nInputs = IHlO[0];
        _nLayers = IHlO.Length - 1;
        _layers = new NeuronsLayer[_nLayers];


        for(Int32 i = 0; i < _nLayers; i ++)
            _layers[i] = new NeuronsLayer(IHlO[i],IHlO[i+1]);

    }

    private double[] activateLayer( Int32 i)
    {

        NeuronsLayer current = _layers[i];
        var iterations = Enumerable.Range(0, current.nNeurons);
        var pquery = from num in iterations.AsParallel() select num;

        numpy.matrixMult_T1(current.inputs, current.weights, ref current.tmp);
        pquery.ForAll((e)=>{ current.z[e]=current.tmp[e][0] + current.biases[e];
                             current.activation[e] = this.Sigmoid(current.z[e]);
                             current.z_prime[e] = this.SigmoidDerivative(current.activation[e]);
                             });
        return current.activation;
    }

    public double[] Feedfordward(double[] inputs)
    {

        _layers[0].inputs[0]=inputs;
        activateLayer(0);

        for (Int32 i = 1; i < _nLayers; i++){
            _layers[i].inputs[0]=_layers[i-1].activation;
            activateLayer(i);
        }
        return Activation;
    }

    public void BackProp(double[] desiredValues)
    {
        
        Int32 i, l;
        NeuronsLayer current = _layers[_nLayers-1]; // we start with the last layer
        double[] delta = new double[current.nNeurons];

        l = current.nNeurons;
        for(i = 0; i < l; i ++)
            delta[i] = (current.activation[i] - desiredValues[i])*current.z_prime[i];
        
        current.nabla_b = delta;
        current.nabla_w = numpy.matrixMult(current.inputs[0], delta);

        //and we work on every previous layer by using the weights of the next layer

        for(i = _nLayers-2; i >=0; i--){
            current = _layers[i];
            delta =numpy.hadamart(numpy.matrixMult(_layers[i+1].weights, delta)[0], current.z_prime);
            current.nabla_b = delta;
            current.nabla_w = numpy.matrixMult(current.inputs[0], delta);
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

    //TODO
    public void pseudoTrain(double[] inputs, double[] desired)
    {
        //we do a feedforward
        Feedfordward(inputs);
        //then calculate some "nablas" for weights and biases
        BackProp(desired);

        //now we adjust weights and biases



    }

    private void cleanNablaTmp(){
        for(Int32 i = 0; i < _nLayers; i++)
            _layers[i].cleanNablaTmp();
    }

    private void adjustWeightsAndBiases(Int32 pIndex, double pSamplesNumber, double pLearningRate){
        NeuronsLayer current = _layers[pIndex];
        Int32 i, j;
        //adjust biases
        for(i = 0; i < current.nNeurons; i++){
            current.nabla_b_tmp[i]+=current.nabla_b[i];
            current.biases[i]-=((pLearningRate/pSamplesNumber)*current.nabla_b_tmp[i]);
        }

        for(i = 0; i < current.nNeurons; i++)
            for(j = 0; j < current.nInput; j ++){
                current.nabla_w_tmp[i][j]+=current.nabla_w[i][j];
                current.weights[i][j]-=((pLearningRate/pSamplesNumber)*current.nabla_w_tmp[i][j]);
            }
    }

    public void pseudoTrainSet(double[][] inputs, double[][] desired)
    {
        cleanNablaTmp();
        
        for (Int32 i = 0; i < inputs.Length; i++){
            //we create a result
            Feedfordward(inputs[i]);
            //we prepare a backpropagation
            BackProp(desired[i]);
            //now we calculate and change biases and weights

            for(Int32 iLayer = 0; iLayer < _nLayers; iLayer++ )
                adjustWeightsAndBiases(iLayer, inputs.Length, this._learningRate);
        }
    }
}


