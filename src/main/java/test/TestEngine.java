package test;

import micrograd.engine.Value;

/**
 * @author sy
 * @date 2022/5/25 20:58
 */
public class TestEngine {

    public static void main(String[] args) {
        testMoreOps();
    }


    public static void testSanityCheck() {
        Value x = new Value(-4.0);
        Value z = x.mul(new Value(2)).add(new Value(2)).add(x);
        Value q = z.relu().add(z.mul(x));
        Value h = (z.mul(z)).relu();
        Value y = h.add(q).add(q.mul(x));

        y.backward();
        System.out.println("x.data -> " + x.data + "; " + "x.grad -> " + x.grad);
        System.out.println("y.data -> " + y.data + "; " + "y.grad -> " + y.grad);
    }

    public static void testMoreOps() {
        Value a = new Value(-4.0);
        Value b = new Value(2.0);
        Value c = a.add(b);
        Value d = a.mul(b).add(b.pow(3));
        c = c.add(c.add(1));
        c = c.add(c.add(1).add(a.neg()));
        d = d.add(d.mul(2).add((b.add(a).relu())));
        d = d.add(d.mul(3).add((b.add(a.neg())).relu()));
        Value e = c.add(d.neg());
        Value f = e.pow(2);
        Value g = f.div(2.0);
        g = g.add(f.rdiv(10.0));

        g.backward();
        System.out.println("a.data -> " + a.data + "; " + "a.grad -> " + a.grad);
        System.out.println("b.data -> " + b.data + "; " + "b.grad -> " + b.grad);
        System.out.println("g.data -> " + g.data + "; " + "g.grad -> " + g.grad);
    }

}
