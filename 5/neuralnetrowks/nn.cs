using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Linq;




struct struNeuron
{
    public double[] weights;
    public double[] errorFeedback;
    public double bias;
    public double z;
    public double output;
}

public interface IActivationClass
{
    public double ActivationFunction(double value);
    public double ActivationDerivative(double value);
}

public class Sigmoid:IActivationClass {
    public double ActivationFunction(double z){
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    private double ActivationDerivative(double s)
    {
        return s * (1 - s);
    }
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
    public double sumSqaresWeights;

    protected IActivationClass activationClass; 
 
    public NeuronsLayer(Int32 pnInputs, Int32 pnNeurons, IActivationClass pAC){
        double tmp = 0;
        this.nInput = pnInputs;
        this.nNeurons = pnNeurons;

        this.activationClass = pAC;

        this.inputs[0] = new double[pnInputs];
        
        this.weights = new double[pnNeurons][];
        this.tmp = new double[pnNeurons][];
        
        tmp = 1.0/Math.Sqrt(pnNeurons);
        for (Int32 i = 0; i < pnNeurons; i ++){
            this.weights[i] = numpy.scalar(numpy.randn1(pnInputs), tmp);
            this.tmp[i] = new double[1] { 0 };
        }

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

    public double[] activate(){
        
        var iterations = Enumerable.Range(0, this.nNeurons);
        var pquery = from num in iterations.AsParallel() select num;

        numpy.matrixMult_T1(this.inputs, this.weights, ref this.tmp);
        pquery.ForAll((e)=>{ this.z[e]=this.tmp[e][0] + this.biases[e];
                             this.activation[e] = this.activationClass.ActivationFunction(this.z[e]); 
                             this.z_prime[e] = this.activationClass.ActivationDerivative(this.activation[e]); 
                             });
        return this.activation;  
    }
}

public class FeatureMapLayer:NeuronsLayer {

    private Int32 strideLength;
    private Int32 inputW, inputH;
    private Int32 recField_edgeSize;

    public FeatureMapLayer(Int32 pInputW, Int32 pInputH, IActivationClass pAC, Int32 pStrideLength, Int32 pRecField_edge_size)
        double tmp = 0;
        
        
        this.nInput = pInputH * pInputW;
        this.nNeurons = (pInputW-pRecField_edge_size+1)*(pInputH-pRecField_edge_size+1);
        this.inputW = pInputW;
        this.inputH = pInputH;
        this.strideLength = pStrideLength;
        this.recField_edgeSize = pRecField_edge_size;

        this.inputs[0] = new double[this.nInput];

        //we have just as many weights as inputs are in the receptive field
        this.weights = new double[1][];
        this.tmp = new double[1][];
        tmp = 1.0/Math.Sqrt(this.nNeurons);        
        this.weights[0] = numpy.scalar(numpy.randn1(pRecField_edge_size*pRecField_edge_size), tmp);

        //we have just ONE bias
        this.biases = numpy.randn1(1);
        this.nabla_b = new double[1];

        //we have as many activations as neurons. 
        this.z = new double[this.nNeurons];
        this.activation = new double[this.nNeurons];
        this.z_prime = new double[this.nNeurons];

        this.activationClass = pAC;
    }

    public double[] activate(){

    }
}





public class NeuralNetwork
{
    private Int32 _nLayers;
    private Int32 _nInputs;
    private NeuronsLayer[] _layers; //first one is the first hidden layer, last one is the output layer 
    private double _learningRate = 0.5;

    public double LearningRate{
        get { return _learningRate;}
        set { _learningRate = value;}
    }

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
        _nInputs = IHlO[0];
        _nLayers = IHlO.Length - 1;
        _layers = new NeuronsLayer[_nLayers];


        for(Int32 i = 0; i < _nLayers; i ++)
            _layers[i] = new NeuronsLayer(IHlO[i],IHlO[i+1]);

    }


    public double[] Feedfordward(double[] inputs)
    {

        _layers[0].inputs[0]=inputs;
        activateLayer(0);

        for (Int32 i = 1; i < _nLayers; i++){
            _layers[i].inputs[0]=_layers[i-1].activation;
            _layers[i].activate();
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
            delta[i] = this.Cost.delta(desiredValues[i], current.activation[i], current.z_prime[i]); //, current.z_prime[i]);
            
        current.nabla_b = delta;
        current.nabla_b_tmp = numpy.add(current.nabla_b, current.nabla_b_tmp);
        current.nabla_w = numpy.matrixMult(current.inputs[0], delta);
        current.nabla_w_tmp = numpy.add(current.nabla_w_tmp, current.nabla_w);

        //and we work on every previous layer by using the weights of the next layer

        for(i = _nLayers-2; i >=0; i--){
            current = _layers[i];
            delta =numpy.hadamart(numpy.matrixMult(_layers[i+1].weights, delta)[0], current.z_prime);
            current.nabla_b = delta;
            current.nabla_b_tmp = numpy.add(current.nabla_b, current.nabla_b_tmp); //we populate temporal arrays by adding for later calculus
            current.nabla_w = numpy.matrixMult(current.inputs[0], delta);
            current.nabla_w_tmp = numpy.add(current.nabla_w_tmp, current.nabla_w); //we populate temporal arrays by adding for later calculus
        }

    }

   

    public void FeedfordwardSet(double[][] inputs, double[][] desired, Int32 samplesToLearn)
    {

        // System.Text.StringBuilder texto = new System.Text.StringBuilder("");
        double[] salida;
        System.Text.StringBuilder[] esperado = new System.Text.StringBuilder[desired.Length];

        for (Int32 i = 0; i < samplesToLearn; i++)
        {
            esperado[i] = new System.Text.StringBuilder("    Esperado: ");
            for (Int32 j = 0; j < desired[i].Length; j++)
                esperado[i].Append(String.Format(" {0:N0}", desired[i][j]));
        }


        Console.SetCursorPosition(0, 0);
        for (Int32 i = 0; i < samplesToLearn; i++)
        {
            salida = Feedfordward(inputs[i]);
            //texto.Clear();
            for (Int32 j = 0; j < salida.Length; j++)
            {
                if (desired[i][j] == 1.0)
                {
                    Console.BackgroundColor = ConsoleColor.DarkGreen;
                    Console.ForegroundColor = ConsoleColor.Gray;
                }
                //texto.Append(String.Format(" {0:N9}", salida[j]));
                Console.Write(String.Format(" {0:N9}", salida[j]));
                Console.BackgroundColor = ConsoleColor.Black;
                Console.ForegroundColor = ConsoleColor.Gray;

            }
            Console.Write("\n");
          //  Console.Write(esperado[i].ToString() + "\n");
        }


        //Console.WriteLine(texto.ToString());
    }


    private void cleanNablaTmp(){
        var iterations = Enumerable.Range(0, _nLayers);
        var pquery = from num in iterations.AsParallel() select num;
        pquery.ForAll((e) => _layers[e].cleanNablaTmp());
    }

    //we modify weights and biases according to the SGD plus a regularization parameter
    private void adjustWeightsAndBiasesSGD(Int32 pIndex, double pSamplesNumber, double pLearningRate, double lambdaRegParam, double totalSamples){
        NeuronsLayer current = _layers[pIndex];
        
        //adjust biases
        double[] sqareWeightsTmp = new double[current.nNeurons]; //temporal for sqare weights summing

        IEnumerable<Int32> iterations = Enumerable.Range(0, current.nNeurons);
        var pquery = from num in iterations.AsParallel() select num;
        pquery.ForAll((i) => {
            sqareWeightsTmp[i] = 0.0;
            current.biases[i] -= ((pLearningRate / pSamplesNumber) * current.nabla_b_tmp[i]);

            for (Int32 j = 0; j < current.nInput; j++)
            {
                current.weights[i][j] = (1.0 - ((pLearningRate*lambdaRegParam)/totalSamples))* current.weights[i][j] - 
                                                ((pLearningRate / pSamplesNumber) * current.nabla_w_tmp[i][j]);
                sqareWeightsTmp[i]+= (current.weights[i][j]*current.weights[i][j]);
            }
        });

        current.sumSqaresWeights = 0;
        for(Int32 i = 0; i < current.nNeurons; i ++)
            current.sumSqaresWeights+=sqareWeightsTmp[i];

    }

    public ICost Cost;
    public void SetCostFunction(ICost p){
        this.Cost = p;
    }
    

    //we train a set of data, we acumulate the nablas and then we
    //adjust weights and biases by using Stocastic Gradient descent
    //
    //
    //@return: the sum of the squares of every weight
    public double pseudoTrainSetSGD(double[][] inputs, double[][] desired, double lambdaRegParam, double totalSamples)
    {
        cleanNablaTmp();
        IEnumerable<Int32> iterations;
        Int32 l = inputs.Length;
        double weightsSquareSum = 0;

        for (Int32 i = 0; i < l; i++){
            //we create a result
            Feedfordward(inputs[i]);
            //we prepare a backpropagation
            BackProp(desired[i]);
            //now we calculate and change biases and weights
        }

        iterations = Enumerable.Range(0, _nLayers);
        var pquery = from num in iterations.AsParallel() select num;
        pquery.ForAll((iLayer) =>adjustWeightsAndBiasesSGD(iLayer, l, this._learningRate, lambdaRegParam, totalSamples));
    
        for(Int32 i = 0; i < _nLayers; i ++)
            weightsSquareSum+= _layers[i].sumSqaresWeights;

        return weightsSquareSum;

    }
}


