L = zeros(T)
for times in range(N):
    for t in range(T):
        sum1 = 0
        H_kt = ones(n)
        for i in G.nodes():
            for j in y[i]:
                #  一个k连接的节点在t时刻没有受邻居传染的概率
                H_kt[i] *= (1 - beta * P0[j])
            # print(i, H_kt[i])
            P1[i] = 1 - ((1 - P0[i]) * H_kt[i] + sigma * P0[i] * H_kt[i])
            sum1 += P1[i]
        P0 = P1.copy()
        L[t] += sum1 / len(G.nodes())

for t in range(T):
    fr.write('%s,%s\n' % (t + 1, L[t] / N))
fr.close()