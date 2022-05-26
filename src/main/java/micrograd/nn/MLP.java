package micrograd.nn;

import micrograd.engine.Value;
import utils.CollectionUtil;

import java.util.List;
import java.util.stream.IntStream;

/**
 * @author sy
 * @date 2022/5/25 21:06
 */
public class MLP extends Module {
    List<Layer> layerList = null;
    public MLP(int nin, List<Integer> nouts) {
        layerList = CollectionUtil.newArrayList();
        List<Integer> sz = CollectionUtil.newArrayList();
        sz.add(nin);
        sz.addAll(nouts);
        // 2 -> 16 -> 16 -> 1
        IntStream.range(0, nouts.size()).boxed().forEach(i -> this.layerList.add(
                new Layer(sz.get(i), sz.get(i + 1), i!=(nouts.size()-1))
        ));
    }

    public List<Value> apply(List<Value> x) {
        for(Layer layer : this.layerList) {
            x = layer.apply(x);
        }
        return x;
    }

    @Override
    public List<Value> parameters() {
        List<Value> parameterList = CollectionUtil.newArrayList();
        for(Layer layer : this.layerList) {
            for(Value p : layer.parameters()) {
                parameterList.add(p);
            }
        }
        return parameterList;
    }

    @Override
    public String toString() {
        List<String> layerStringList = CollectionUtil.newArrayList();
        for(Layer layer: this.layerList) {
            layerStringList.add(layer.toString());
        }
        return "MLP of [" + String.join(",", layerStringList) + "]";
    }

}


