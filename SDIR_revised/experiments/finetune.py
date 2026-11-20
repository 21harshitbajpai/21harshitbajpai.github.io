import sys, time, torch
import train_cnn as T
model = torch.load('cnn_mse.pt', weights_only=False)
opt = torch.optim.Adam(model.parameters(), 1e-4)
n = int(sys.argv[1]); t0 = time.time()
for it in range(n + 1):
    y, noise = T.batch(16, 48)
    mse = ((model(y) - noise) ** 2).mean()
    yp, _ = T.batch(4, 96)
    lm = T.lam_min_sym(model, yp)
    pen = (torch.relu(0.02 - lm) ** 2).mean()
    loss = mse + 10.0 * pen
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 25 == 0:
        print(f'pen {it} mse {mse.item():.2e} lam_min {lm.min().item():+.3f} {time.time()-t0:.0f}s', flush=True)
    if it % 50 == 0 and it > 0:
        torch.save(model, 'cnn.pt')
print('done', flush=True)
