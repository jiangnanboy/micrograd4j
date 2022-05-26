package micrograd.nn;

import micrograd.engine.Value;
import utils.CollectionUtil;

import java.util.List;
import java.util.stream.IntStream;

/**
 * @author sy
 * @date 2022/5/25 22:06
 */
public class Neuron extends Module {
    List<Value> w;
    Value b;
    boolean nonlin;

    public Neuron (int nin) {
        this(nin, true);
    }

    public Neuron(int nin, boolean nonlin) {
        this.w = CollectionUtil.newArrayList();
        IntStream.range(0, nin).boxed().forEach(i -> this.w.add(new Value(-1 + (float)Math.random() * 2)));
        this.b = new Value(0);
        this.nonlin = nonlin;
    }

    public Value apply(List<Value> x) {
        assert this.w.size() != x.size() : "参数长度不一致！" ;

        List<Value> mulListValue = CollectionUtil.newArrayList();
        for(int index=0; index<this.w.size(); index++) {
            mulListValue.add(this.w.get(index).mul(x.get(index)));
        }
        Value act = null;
        act = mulListValue.get(0);
        for(Value value : mulListValue) {
            act = act.add(value);
        }
        act = act.add(this.b);
        if(this.nonlin) {
            return act.relu();
        } else {
            return act;
        }
    }

    @Override
    public List<Value> parameters() {
        List<Value> parameterList = CollectionUtil.newArrayList(this.w);
        parameterList.add(this.b);
        return parameterList;
    }

    @Override
    public String toString() {
        if(this.nonlin) {
            return "ReLuNeuron(" + this.w.size() + ")";
        } else {
            return "LinearNeuron(" + this.w.size() + ")";
        }
    }

}
