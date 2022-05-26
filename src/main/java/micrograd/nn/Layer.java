package micrograd.nn;

import micrograd.engine.Value;
import utils.CollectionUtil;

import java.util.List;
import java.util.stream.IntStream;

/**
 * @author sy
 * @date 2022/5/25 21:06
 */
public class Layer extends Module {
    List<Neuron> neuronList = null;

    public Layer(int nin, int nout, boolean nonlin) {
        this.neuronList = CollectionUtil.newArrayList();
        IntStream.range(0, nout).boxed().forEach(i->this.neuronList.add(new Neuron(nin, nonlin)));
    }

    public List<Value> apply(List<Value> x) {
        List<Value> outNeuronList = CollectionUtil.newArrayList();
        for(Neuron n : this.neuronList) {
            outNeuronList.add(n.apply(x));
        }
//        return outNeuronList.size()==1?outNeuronList.get(0):outNeuronList;
        return outNeuronList;
    }

    @Override
    public List<Value> parameters() {
        List<Value> parameterList = CollectionUtil.newArrayList();
        for(Neuron n : this.neuronList) {
            for(Value p : n.parameters()) {
                parameterList.add(p);
            }
        }
        return parameterList;
    }

    @Override
    public String toString() {
        List<String> layerStringList = CollectionUtil.newArrayList();
        for(Neuron n: this.neuronList) {
            layerStringList.add(n.toString());
        }
        return "Layer of [" + String.join(",", layerStringList) + "]";
    }

}
