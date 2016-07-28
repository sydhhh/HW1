function K = poly(x,c,d)
    
    K = (linear(x)+c).^d;
    