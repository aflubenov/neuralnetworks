using System;

public interface ICost{
    double cost(double[][] desired, double[][] inputs, NeuralNetwork n);
    double cost(double[][] desired, double[][] activations);
    double delta(double desired, double activation, double zPrime);
}

public class Quadratic: ICost{

    public double cost(double[][] desired, double[][] inputs, NeuralNetwork n){
        Int32 i = 0, l = inputs.Length;
        double[][] activations = new double[l][];

        for(i = 0; i < l; i ++)
            activations[i] = n.Feedfordward(inputs[i]);

        return this.cost(desired, activations);
    }

    public double cost(double[][] desired, double[][] activations){
        Int32 i = 0, li = activations.Length;
        double acumSum = 0.0;

        for(i = 0; i < li; i ++)
                acumSum+=numpy.absSqr(numpy.add(desired[i],numpy.scalar(activations[i],-1.0)));

        return (1.0/(2.0*(double)li))*acumSum;
    }

    public double delta(double desired, double activation, double zPrime){
        return (activation - desired)*zPrime;
    }
}

public class CrossEntropy:ICost{

    public double cost(double[][] desired, double[][] inputs, NeuralNetwork n){
        Int32 i = 0, l = inputs.Length;
        double[][] activations = new double[l][];

        for(i = 0; i < l; i ++)
            activations[i] = n.Feedfordward(inputs[i]);

        return this.cost(desired, activations);
    }

    public double cost(double[][] desired, double[][] activations){
        Int32 i = 0, li = activations.Length, 
              j=0, lj = activations[0].Length;
        double acumSum = 0.0;

              
        for(i = 0; i < li; i ++)
            for(j = 0; j < lj; j ++)
                acumSum+= ((-1.0*desired[i][j]*Math.Log(activations[i][j]))-(1.0 - desired[i][j])*Math.Log(1.0 - activations[i][j] ));

        return (1.0/(double)li)*acumSum;
    }

    public double delta(double desired, double activation, double zPrime){
        zPrime = zPrime+1;
        return activation - desired;
    }
}