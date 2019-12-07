import numpy as np
import cvxpy as cp

build_problem_theta = False
pThetaT, pZT, pGradT, thetaOptT, pbThetaT = None, None, None, None, None
build_problem_mu = False
pThetaM, pZM, pFeatM, pGammaM, thetaOptM, pbMu = None, None, None, None, None, None
def build_theta_optim_pb(dimTheta , Dval):
	paramTheta = cp.Parameter((dimTheta,1))
	paramGrad = cp.Parameter((dimTheta,1))
	paramZ = cp.Parameter(shape=(dimTheta,dimTheta), PSD=True)
	theta = cp.Variable((dimTheta,1))
	pbConstr = list()
	pbConstr.append(cp.norm(theta) <= Dval)
	costFun = 0.5 * cp.quad_form(theta - paramTheta , paramZ) + theta.T * paramGrad
	problem = cp.Problem(cp.Minimize(costFun), pbConstr)
	return paramTheta, paramZ, paramGrad, theta, problem

def build_mu_optim_pb(dimTheta):
	paramTheta = cp.Parameter((dimTheta,1))
	paramFeat = cp.Parameter((dimTheta,1))
	paramZ = cp.Parameter(shape=(dimTheta,dimTheta), PSD=True)
	paramGamma = cp.Parameter(nonneg=True)
	theta = cp.Variable((dimTheta,1))
	pbConstr = list()
	pbConstr.append(cp.quad_form(theta - paramTheta, paramZ) <= paramGamma)
	costFun = theta.T * paramFeat
	problem = cp.Problem(cp.Maximize(costFun), pbConstr)
	return paramTheta, paramZ, paramFeat, paramGamma, theta, problem

def compute_next_theta_hat(theta_hat, D, Z, grad, verb=True):
	global build_problem_theta, pThetaT, pZT, pGradT, thetaOptT, pbThetaT
	if not build_problem_theta:
		print ("Create theta problem")
		pThetaT, pZT, pGradT, thetaOptT, pbThetaT = build_theta_optim_pb(theta_hat.shape[0] , D)
		build_problem_theta = True
	pThetaT.value = theta_hat
	pZT.value = Z
	pGradT.value = grad
	pbThetaT.solve(solver=cp.GUROBI, verbose=verb)
	return thetaOptT.value

def compute_estimate_reward(theta_hat, Z, gamma, xFeat, verb=True):
	global build_problem_mu, pThetaM, pZM, pFeatM, pGammaM, thetaOptM, pbMu
	if not build_problem_mu:
		print ("Create Mu problem")
		pThetaM, pZM, pFeatM, pGammaM, thetaOptM, pbMu = build_mu_optim_pb(theta_hat.shape[0])
		build_problem_mu = True
	pThetaM.value = theta_hat
	pZM.value = Z
	pFeatM.value = xFeat
	pGammaM.value = gamma
	costVal = pbMu.solve(solver=cp.GUROBI, verbose=verb)
	return costVal