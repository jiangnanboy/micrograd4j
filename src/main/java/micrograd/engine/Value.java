package micrograd.engine;

import utils.CollectionUtil;

import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * stores a single scalar value and its gradient
 * @author sy
 * @date 2022/5/24 21:12
 */
public class Value {
    public double data;
    public double grad;
    IBackward backward=null;
    Set<Value> _prev;
    String _op;

    public Value(double data) {
        this(data, Collections.EMPTY_LIST, "");
    }

    public Value(double data, List<Value> _children) {
        this(data, _children, "");
    }

    public Value(double data, List<Value> _children, String _op) {
        this.data = data;
        this.grad = 0;
        // internal variables used for autograd graph construction
        this._prev = CollectionUtil.newHashset(_children);
        // the op that produced this node, for graphviz / debugging / etc
        this._op = _op;
    }

    /**
     * @param otherValue
     * @return
     */
    public Value add(Value otherValue) {
        List<Value> _children = CollectionUtil.newArrayList();
        _children.add(this);
        _children.add(otherValue);
        Value outValue = new Value(this.data + otherValue.data, _children, "+");

        class Backward implements IBackward {
            @Override
            public void _backward() {
                grad += outValue.grad;
                otherValue.grad += outValue.grad;
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    /**
     * @param otherValue
     * @return
     */
    public Value mul(Value otherValue) {
        List<Value> _children = CollectionUtil.newArrayList();
        _children.add(this);
        _children.add(otherValue);
        Value outValue = new Value(this.data * otherValue.data, _children, "*");
        class Backward implements IBackward {
            @Override
            public void _backward() {
                grad += otherValue.data * outValue.grad;
                otherValue.grad += data * outValue.grad;
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    /**
     * @param otherValue
     * @return
     */
    public Value pow(double otherValue) {
        List<Value> _children = CollectionUtil.newArrayList();
        _children.add(this);
        Value outValue = new Value(Math.pow(this.data, otherValue), _children, "**");
        class Backward implements IBackward {
            @Override
            public void _backward() {
                grad += (otherValue * Math.pow(data, (otherValue - 1))) * outValue.grad;
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    /**
     * @return
     */
    public Value relu() {
        List<Value> _children = CollectionUtil.newArrayList();
        _children.add(this);
        Value outValue;
        if(this.data < 0) {
            outValue = new Value(0, _children, "ReLu");
        } else {
            outValue = new Value(this.data, _children, "ReLu");
        }
        class Backward implements IBackward {
            @Override
            public void _backward() {
                if(outValue.data > 0) {
                    grad += 1 * outValue.grad;
                } else {
                    grad += 0 * outValue.grad;
                }
            }
        }
        outValue.backward = new Backward();
        return outValue;
    }

    /**
     * @param value
     */
    public void buildTopo(Value value, List<Value> topoList, Set<Value> visitedSet) {
        if(!visitedSet.contains(value)) {
            visitedSet.add(value);
            for(Value child : value._prev) {
                buildTopo(child, topoList, visitedSet);
            }
            topoList.add(value);
        }
    }

    /**
     *
     */
    public void backward() {
        // topological order all of the children in the graph
        List<Value> topoList = CollectionUtil.newArrayList();
        Set<Value> visitedSet = CollectionUtil.newHashset();
        buildTopo(this, topoList, visitedSet);
        // go one variable at a time and apple the chain rule to get its gradient
        this.grad = 1;
        Collections.reverse(topoList);
        for(Value value : topoList) {
            if(null != value.backward) {
                value.backward._backward();
            }
        }
    }

    public Value neg() {
        return this.mul(new Value(-1));
    }

    public Value add(double other) {
        return this.add(new Value(other));
    }

    public Value sub(double other) {
        other = other * (-1);
        return this.add(new Value(other));
    }

    public Value mul(double other) {
        return this.mul(new Value(other));
    }

    public Value div(double other) {
        return this.mul(new Value(other).pow(-1));
    }

    public Value rdiv(double other) {
        return new Value(other).mul(this.pow(-1));
    }

    @Override
    public String toString() {
        return "Value(data={"+ this.data +"}, grad={" +this.grad + "})";
    }

}

