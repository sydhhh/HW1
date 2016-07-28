function k = my_gaussian(x1, x2, sigma)
    x_y         =   x1-x2;
    Normx_y    =   x_y*x_y';
    k         =   exp(-Normx_y/(2*sigma^2));
end