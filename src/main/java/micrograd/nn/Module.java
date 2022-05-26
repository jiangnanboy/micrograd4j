package micrograd.nn;

import micrograd.engine.Value;

import java.util.Collections;
import java.util.List;

/**
 * @author sy
 * @date 2022/5/25 22:06
 */
public class Module {

    public void zeroGrad() {
        for(Value p : this.parameters()) {
            p.grad = 0;
        }
    }

    public List<Value> parameters() {
        return Collections.EMPTY_LIST;
    }

}
