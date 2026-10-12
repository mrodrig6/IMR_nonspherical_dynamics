function [idxT, idxV, idxep, idxepd, blockSize] = f_get_indicies(forcedep, mod, xN, i, rot)

if rot == "irr"
    blockSize = 2;
    base = (i-1)*blockSize;
    idxep = base + 1;
    idxepd = base + 2;
    idxT = [];
    idxV  = [];
elseif forcedep == 'T' && rot == "rot"
    if mod == "me"
        blockSize = 2*xN;
        base = (i-1)*blockSize;
        % ---- indices ----
        idxT   = base + (1:xN);
        idxV  = base + xN + (1:xN);
        idxep  = [];
        idxepd = [];
    elseif mod == "Pros" && rot == "rot"
        blockSize = xN;
        base = (i-1)*blockSize;
        % ---- indices ----
        idxT   = base + (1:xN);
        idxV  = [];
        idxep  = [];
        idxepd = [];
    end
elseif forcedep ~= 'T' && rot == "rot"
    if mod == "me"
        blockSize = 2*xN+2;
        base = (i-1)*blockSize;
        % ---- indices ----
        idxT   = base + (1:xN);
        idxV  = base + xN + (1:xN);
        idxep  = base + 2*xN + 1;
        idxepd = base + 2*xN + 2;
    elseif mod == "Pros" && rot == "rot"
        blockSize = xN+2;
        base = (i-1)*blockSize;
        % ---- indices ----
        idxT   = base + (1:xN);
        idxep  = base + xN + 1;
        idxepd = base + xN + 2;
        idxV  = [];
    end
end
end