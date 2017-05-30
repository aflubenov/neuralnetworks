using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Linq;





public abstract class NNLayer{
    public Int32 nInput = 0;
    public Int32 nNeurons = 0;
    protected double[][] inputs = new double[1][]; //1 x number of inputs
    public double[][] weights; //number of neurons x number of inputs
    public double[] biases; //number of neurons 
    public double[][] tmp; //number of neurons x 0
    public double[] z; 
    public double[] activation; //number of neurons
    //-------- helper
    public double[] z_prime; //number of neurons
    public double[][] nabla_w; //number of neurons x number of inputs
    public double[][] nabla_w_tmp;

    public double[] nabla_b; //number of neurons
    public double[] nabla_b_tmp;
    public double sumSqaresWeights;

    protected IActivationClass activationClass; 
 
    
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

    public abstract double[] activate();
    public abstract void setInputs(double[] p);
    public abstract double[] BackProp(double[] desiredValues, ICost Cost, double[] delta = null, NNLayer nextLayer = null );
}

public class NeuronsLayer:NNLayer{

    public override void setInputs(double[] p) {
        this.inputs[0] = p;
    }

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


    public override double[] BackProp(double[] desiredValues, ICost Cost, double[] delta = null, NNLayer nextLayer = null )
    {
        
        Int32 i;

        if(delta == null){

            delta = new double[this.nNeurons];
            for(i = 0; i < this.nNeurons; i ++)
                delta[i] = Cost.delta(desiredValues[i], this.activation[i], this.z_prime[i]);
        } else  {
            delta = numpy.hadamart(numpy.matrixMult(nextLayer.weights, delta)[0], this.z_prime); 
        }
        this.nabla_b = delta;
        this.nabla_b_tmp = numpy.add(this.nabla_b, this.nabla_b_tmp); //we populate temporal arrays by adding for later calculus
        this.nabla_w = numpy.matrixMult(this.inputs[0], delta);
        this.nabla_w_tmp = numpy.add(this.nabla_w_tmp, this.nabla_w); //we populate temporal arrays by adding for later calculus

        //and we work on every previous layer by using the weights of the next layer

        return delta;
    }



    public override double[] activate(){
        
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

public class FeatureMapLayer:NNLayer {

    protected Int32 strideLength;
    protected Int32 inputW, inputH;
    protected Int32 recField_edgeSize;
    protected Int32 outputW, outputH;

    public Int32 OutputW{
        get{
            return outputW; 
        }
    }

    public Int32 OutputH{
        get{
            return outputH;
        }
    }

    public override void setInputs(double[] p) {

        if(p.Length < this.inputW * this.inputH)
            throw new Exception("input smaller than expected");

        this.inputs[0] = p;
    }

    public FeatureMapLayer(Int32 pInputW, Int32 pInputH, IActivationClass pAC, Int32 pStrideLength, Int32 pRecField_edge_size){
        double tmp = 0;
        
        
        this.nInput = pInputH * pInputW;
        //how many times the recceptive field square fits into the
        // input?, (inputW-recFieldSize)/strideLength +1 ... same as height
        this.outputW = ((pInputW-pRecField_edge_size)/pStrideLength)+1;
        this.outputH = ((pInputH-pRecField_edge_size)/pStrideLength)+1;

        this.nNeurons = outputW*outputH;
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

    public override double[] activate(){
        //we have to walk over the inputs VECTOR as an rectangular matrix
        //so the input in the position "x,y" is y*width+x
        //and we have recField_edgeSize**2 inputs that goes from
        //[x,y] to [x+recField_EdgeSize, y+recField_EdgeSize]

        Int32 ox, oy;
        Int32 ix, iy;

        double acumSum = 0.0;
        Int32 i, j, n;

        //there is a biyective relation between the output matrix and 
        //the input one and the stride length

        //for every neuron in the output vector, we infer the x,y coordinates of
        // the "ouput rectangular matrix" and then we infer the starting 
        // coordinates of the receptive field in the input matrix
        for(n = 0; n < this.nNeurons; n ++){
            //we get the coordinates for the output rectangle, based on
            // that, we get the coordinates in the input. 
            oy = n/this.outputW;
            ox = n % this.outputW;
            ix = ox*this.strideLength;
            iy = oy*this.strideLength;
            
            acumSum = this.biases[0];
            for(i = 0; i < this.recField_edgeSize; i++ )
                for(j = 0; j < this.recField_edgeSize; j++)
                    acumSum += this.inputs[0][(iy+j) *this.inputW + ix + i] * this.weights[0][j*this.recField_edgeSize+i];
            
            this.z[n] = acumSum;
            this.activation[n] = this.activationClass.ActivationFunction(this.z[n]); 
            this.z_prime[n] = this.activationClass.ActivationDerivative(this.activation[n]);
        }

        return this.activation;
    }

    public override double[] BackProp(double[] desiredValues, ICost Cost, double[] delta = null, NNLayer nextLayer = null )
    {
        
        Int32 i;

        if(delta == null){
            throw new Exception("we are not expecting a feature map layer be the last one :-( ... yet");
            //
            //            delta = new double[this.nNeurons];
            //            for(i = 0; i < this.nNeurons; i ++)
            //                delta[i] = Cost.delta(desiredValues[i], this.activation[i], this.z_prime[i]);
        } else  {
            delta = numpy.hadamart(numpy.matrixMult(nextLayer.weights, delta)[0], this.z_prime); 
        }
        this.nabla_b = delta;
        this.nabla_b_tmp = numpy.add(this.nabla_b, this.nabla_b_tmp); //we populate temporal arrays by adding for later calculus
        this.nabla_w = numpy.matrixMult(this.inputs[0], delta);
        this.nabla_w_tmp = numpy.add(this.nabla_w_tmp, this.nabla_w); //we populate temporal arrays by adding for later calculus

        //and we work on every previous layer by using the weights of the next layer

        return delta;
    }
}

public class PoolingLayer:FeatureMapLayer {

    public PoolingLayer(Int32 pInputW, Int32 pInputH, IActivationClass pAC, Int32 pStrideLength, Int32 pRecField_edge_size):base( pInputW,  pInputH,  pAC,  pStrideLength,  pRecField_edge_size){

    }

    public override double[] activate(){
        //we have to walk over the inputs VECTOR as an rectangular matrix
        //so the input in the position "x,y" is y*width+x
        //and we have recField_edgeSize**2 inputs that goes from
        //[x,y] to [x+recField_EdgeSize, y+recField_EdgeSize]

        Int32 ox, oy;
        Int32 ix, iy;

        double maxValue = Double.MinValue;
        double tmp;
        Int32 i, j, n;

        //there is a biyective relation between the output matrix and 
        //the input one and the stride length

        //for every neuron in the output vector, we infer the x,y coordinates of
        // the "ouput rectangular matrix" and then we infer the starting 
        // coordinates of the receptive field in the input matrix
        for(n = 0; n < this.nNeurons; n ++){
            //we get the coordinates for the output rectangle, based on
            // that, we get the coordinates in the input. 
            oy = n/this.outputW;
            ox = n % this.outputW;
            ix = ox*this.strideLength;
            iy = oy*this.strideLength;
            
            for(i = 0; i < this.recField_edgeSize; i++ )
                for(j = 0; j < this.recField_edgeSize; j++){
                    tmp = this.inputs[0][(iy+j)*this.inputW+ix+i];
                    maxValue = tmp > maxValue?tmp:maxValue; //<--- by the moment we get just the max value, later we are gonna generalize this
                    
                }
            
            this.z[n] = maxValue;
            this.activation[n] = maxValue; 
            this.z_prime[n] = maxValue;
        }

        return this.activation;
    }

    public override double[] BackProp(double[] desiredValues, ICost Cost, double[] delta = null, NNLayer nextLayer = null ){
        
        //pooling layers should not have backpropagation :-) 
        return null;
    }
    
}

public class ConvolutionLayer:NNLayer {

    private FeatureMapLayer[] _fml;
    private PoolingLayer[] _pl; 
    private Int32 _nLayers; 
    
    public override void setInputs(double[] p){
        for(Int32 i = 0; i < _nLayers; i++)
            this._fml[i].setInputs(p);
    }

    public ConvolutionLayer(Int32 pInputW, Int32 pInputH, IActivationClass pAC, Int32 pStrideLength, Int32 pRecField_edge_size, Int32 pRecField_Pooling_Layer_edge_size, Int32 pHowmany){
        this._fml = new FeatureMapLayer[pHowmany];
        this._pl =new PoolingLayer[pHowmany];
        this._nLayers = pHowmany;

        for(Int32 i = 0; i < pHowmany; i ++){
            this._fml[i] = new FeatureMapLayer(pInputW, pInputH, pAC, pStrideLength, pRecField_edge_size);
            this._pl[i] = new PoolingLayer(this._fml[i].OutputW, this._fml[i].OutputH, pAC, pRecField_Pooling_Layer_edge_size, pRecField_Pooling_Layer_edge_size);
        }

        this.activation = new double[this._pl[0].OutputW*this._pl[0].OutputH*pHowmany];

    }

    public override double[] activate(){
        double[] tmp;
        Int32 oLength; 

        for(Int32 i = 0; i < _nLayers; i ++){
            this._pl[i].setInputs(this._fml[i].activate());
            tmp =  this._pl[i].activate();

            oLength = tmp.Length;
            for(Int32 j = 0; j < oLength; j ++)
                this.activation[i * oLength + j] = tmp[j];
        }
        //we return one array concatenated with all the activations from every pooling layer, sorted in 
        //the same way as the pooling layers are.
        return this.activation;
        
    }

    public override double[] BackProp(double[] desiredValues, ICost Cost, double[] delta = null, NNLayer nextLayer = null )
    {

        //backpropagating on a convolutional layer mean: 
        // 0.- "delta" always has as many values as NEURONS in the NEXT LAYER
        // 0.5 - the NEW 'delta' calculation will have as many values as NEURONS WE HAVE NOW
        // 1.- weights in the next layer are as many weights as the sum of all the outputs in pooling layer
        // 2.- since output is an array with the activation of all the pooling layers sortened, what
        //      we have in the new delta is a sortened array of every activation of all the pooling layers. 
        // 3.- we can SPLIT the delta into as many arrays as pooling layers (mini deltas),
        // 4.- every activation's value of one pooling layer IS A VALUE OF ITS FEATURE MAP LAYER (unless with the pooling layer we are using now that doesn't change values)
        // 5.- so (by the monent) we can ignore the pooling layer and sum every value of each mini-delta simulating that we did "as many iterations as receptive fields we have")   
        // 6.- What happen if the "next layer" is a convolution layer???? WE DON'T KNOW!!!!, by the moment we won't use 2 convolution layers :'( 

        Int32 i, j;
        double[][] deltas;

        if(delta == null){
            throw new Exception("we are not expecting a CONVOLUTION layer be the last one :-( ... yet");
//
//            delta = new double[this.nNeurons];
//            for(i = 0; i < this.nNeurons; i ++)
//                delta[i] = Cost.delta(desiredValues[i], this.activation[i], this.z_prime[i]);
        } else  {
            /////////////////////////delta = numpy.hadamart(numpy.matrixMult(nextLayer.weights, delta)[0], this.z_prime);
            
            delta = numpy.matrixMult(nextLayer.weights, delta)[0]; //in further steps we have to hadamart by z_prime. 

            //we should split the "delta" in as many as pooling layers we have. 
            deltas = new double[_nLayers][]; //an array of _nLayers * sizeof(output of pooling layer)
            for(i = 0; i < _nLayers; i ++) {
                deltas[i] = new double[this._pl[i].nNeurons];

                for(j = 0; j < this._pl[i].nNeurons; j ++)
                    deltas[i][j] = delta[i*this._pl[i].nNeurons+j];
            }
                
        }
        this.nabla_b = delta;
        this.nabla_b_tmp = numpy.add(this.nabla_b, this.nabla_b_tmp); //we populate temporal arrays by adding for later calculus
        this.nabla_w = numpy.matrixMult(this.inputs[0], delta);
        this.nabla_w_tmp = numpy.add(this.nabla_w_tmp, this.nabla_w); //we populate temporal arrays by adding for later calculus

        //and we work on every previous layer by using the weights of the next layer

        return delta;
    }

    
}