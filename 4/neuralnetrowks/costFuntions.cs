using System;


public class Quadratic{

    public static double cost(double[][] desired, double[][] inputs, NeuralNetwork n){
        Int32 i = 0, l = inputs.Length;
        double[][] activations = new double[l][];

        for(i = 0; i < l; i ++)
            activations[i] = n.Feedfordward(inputs[i]);

        return CrossEntropy.cost(desired, activations);
    }

    public static double cost(double[][] desired, double[][] activations){
        Int32 i = 0, li = activations.Length;
        double acumSum = 0.0;

        for(i = 0; i < li; i ++)
                acumSum+=numpy.absSqr(numpy.add(desired[i],numpy.scalar(activations[i],-1.0)));

        return (1.0/(2.0*(double)li))*acumSum;
    }

    public static double delta(double desired, double activation, double zPrime){
        return (activation - desired)*zPrime;
    }
}

public class CrossEntropy{

    public static double cost(double[][] desired, double[][] inputs, NeuralNetwork n){
        Int32 i = 0, l = inputs.Length;
        double[][] activations = new double[l][];

        for(i = 0; i < l; i ++)
            activations[i] = n.Feedfordward(inputs[i]);

        return CrossEntropy.cost(desired, activations);
    }

    public static double cost(double[][] desired, double[][] activations){
        Int32 i = 0, li = activations.Length, 
              j=0, lj = activations[0].Length;
        double acumSum = 0.0;

              
        for(i = 0; i < li; i ++)
            for(j = 0; j < lj; j ++)
                acumSum+= ((-1.0*desired[i][j]*Math.Log(activations[i][j]))-(1.0 - desired[i][j])*Math.Log(1.0 - activations[i][j] ));

        return (1.0/(double)li)*acumSum;
    }

    public static double delta(double desired, double activation){
        return activation - desired;
    }
}