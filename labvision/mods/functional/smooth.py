def smooth(inputs, weight=0.7):

    inputs = [x[-1] if type(x) is tuple else x for x in inputs.history]  # for reading accuracy_module history

    for i in range(int(10*weight)+1):
        ret = []
        for idx, x in enumerate(inputs):
            if idx == 0 or idx == len(inputs)-1:
                ret.append(x)
            else:
                ret.append((inputs[idx-1]+x+inputs[idx+1])/3*weight+x*(1-weight))
        inputs = ret
    return ret
