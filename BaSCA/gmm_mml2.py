#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:44:27 2017

@author: abriosi
"""

from sklearn.base import TransformerMixin
import numpy as np
from scipy.stats import multivariate_normal

class GmmMml2(TransformerMixin):

    def __init__(self,kmin=1,
                 kmax=25,
                 regularize=1e-6,
                 threshold=1e-5,
                 covoption=0,
                 max_iters=100,
                 plots=False):

        self.kmin=kmin
        self.kmax=kmax
        self.regularize=regularize
        self.th=threshold
        self.covoption=covoption
        self.max_iters=max_iters
        self.check_plot=plots

    def _posterior_probability(self,y,estmu,estcov,i):
        try:
            return multivariate_normal.pdf(y, estmu[i], estcov[:,:,i], allow_singular=False)
        except Exception as inst:
            raise Exception(f'y: {y}, estmu[i]: {estmu[i]}, estcov[:,:,i]: {estcov[:,:,i]} Possible singular matrix detected. Try adding (more) regularization')


    def fit(self, X, y=None, verb=False):
        y=np.array(X)

        dl=[]
        npoints=y.shape[0]
        # print(f'npoint: { npoints}')
        dimens=y.shape[1]

        if self.covoption==0:
            npars = (dimens + dimens*(dimens+1)/2)
        elif self.covoption==1:
            npars = 2*dimens
        elif self.covoption==2:
            npars = dimens
        elif self.covoption==3:
            npars = dimens
        else:
            npars = (dimens + dimens*(dimens+1)/2)

        nparsover2 = npars / 2

        k = self.kmax

        indic = np.zeros((k,npoints))
        randindex = np.random.randint(0,npoints,npoints)
        randindex = np.random.choice(randindex,k)
        estmu = y[randindex]

        estpp = (1/float(k))*np.ones((1,k))

        if dimens > 1:
            globcov = np.cov(y,rowvar=False)
        else:
            globcov = np.array([np.array([np.cov(y,rowvar=False)])])

        estcov=np.empty(globcov.shape+(self.kmax,))
        for i in range(k):
            estcov[:,:,i]=np.diag((np.diag(np.ones((dimens,dimens))*np.max(np.diag(globcov/10)))))

        # if self.check_plot== True:
        #     self._plot_graph(estcov,estmu,k,y,'Random Guassian Initialization')

        semi_indic=np.empty((k,y.shape[0]))
        for i in range(k):

            min_eig = np.min(np.real(np.linalg.eigvals(estcov[:,:,i])))
            if min_eig < 0:
                estcov[:,:,i] -= 10*min_eig * np.eye(*estcov[:,:,i].shape)
            semi_indic[i,:]=self._posterior_probability(y,estmu,estcov,i)
            indic[i,:]=semi_indic[i,:]*estpp[:,i]

        countf = 0
        loglike=[]
        kappas=[]

        loglike.append(np.sum(np.log(np.sum(np.finfo(np.float64).tiny+indic,axis=0))))
        dlength = -loglike[countf] + (nparsover2*np.sum(np.log(estpp))) + (nparsover2 + 0.5)*k*np.log(npoints)
        dl.append(dlength)
        kappas.append(k)

        transitions1 = []
        transitions2 = []

        mindl = dl[countf]
        self.bestmu = estmu.copy()
        self.bestcov = estcov.copy()
        # print(self.bestcov.shape)
        # for i in range(self.bestcov.shape[2]):
        #     print(f'first: {self.bestcov[:,:,i]}')

        k_cont = True

        iteration=0

        while k_cont==True:
            cont=True

            while cont == True and iteration < self.max_iters:
                # if verb==True:
                #     print('k='+str(k)+' minestpp='+str(np.min(estpp)))
                comp = 0
                while comp < k:
                    # print(f'comp: {comp}')
                    # for a in range(self.bestcov.shape[2]):
                        # print(f'while bestcov: {self.bestcov[:,:,a]}')
                    indic = np.zeros((k,npoints))
                    for i in range(k):
                        indic[i,:]=semi_indic[i,:]*estpp[:,i]
                    normindic = np.divide(indic,(np.finfo(np.float64).tiny+np.kron(np.ones((k,1)),np.sum(indic,axis=0))))
                    normalize = 1/np.sum(normindic[comp,:],axis=0)
                    aux=np.multiply(np.kron(normindic[comp,:],np.ones((dimens,1))),y.T)
                    estmu[comp,:] = normalize*np.sum(aux,axis=-1)

                    if self.covoption == 0:
                        estcov[:,:,comp]=normalize*aux.dot(y) - estmu[comp,:][:,np.newaxis]*estmu[comp,:][:,np.newaxis].T + self.regularize*np.identity(dimens)
                        # print(f'130gyou: {estcov[:,:,comp]}')
                    else:
                        raise NameError('Not implemented covoption > 0')

                    estpp[:,comp] = np.max(np.sum(normindic[comp,:])-nparsover2,axis=0)/npoints
                    estpp = estpp/np.sum(estpp)
                    killed = 0
                    if estpp[:,comp]<=0:
                        killed=1
                        transitions1.append(countf)
                        estmu = np.delete(estmu, comp, axis=0)
                        estcov = np.delete(estcov, comp, axis=-1)
                        # print('estpp[:,comp]<=0\n')
                        estpp = np.delete(estpp, comp, axis=-1)
                        semi_indic=np.delete(semi_indic, comp, axis=0)
                        k=k-1

                    if killed==0:
                        #kokokousinn
                        min_eig = np.min(np.real(np.linalg.eigvals(estcov[:,:,comp])))
                        if min_eig < 0:
                            # print(f'before update cov: {estcov[:,:,comp]}')
                            # print(f'kari bestcov: {self.bestcov[:,:,comp]}')
                            estcov[:,:,comp] -= 10*min_eig * np.eye(*estcov[:,:,comp].shape)
                            # print(min_eig, comp)
                            # print(f'update cov: {estcov[:,:,comp]}')
                            # print(f'kari bestcov: {self.bestcov[:,:,comp]}')
                        semi_indic[comp,:]=self._posterior_probability(y,estmu,estcov,comp)
                        comp+=1

                countf = countf + 1

                indic = np.zeros((k,npoints))
                semi_indic = np.empty((k,y.shape[0]))
                for i in range(k):
                    min_eig = np.min(np.real(np.linalg.eigvals(estcov[:,:,i])))
                    if min_eig < 0:
                        estcov[:,:,i] -= 10*min_eig * np.eye(*estcov[:,:,i].shape)
                        # print(f'166gyou: {estcov[:,:,i]}')
                    semi_indic[i,:]=self._posterior_probability(y,estmu,estcov,i)
                    indic[i,:]=semi_indic[i,:]*estpp[:,i]

                if k != 1:
                    loglike.append(np.sum(np.log(np.finfo(np.float64).tiny+np.sum(indic,axis=0))))
                else:
                    loglike.append(np.sum(np.log(np.finfo(np.float64).tiny+indic)))

                dlength = -loglike[countf] + (nparsover2*np.sum(np.log(estpp))) + (nparsover2 + 0.5)*k*np.log(npoints)
                dl.append(dlength)
                kappas.append(k)
                deltlike = loglike[countf] - loglike[countf-1]

                if verb==True:
                    toprint=np.abs(deltlike/loglike[countf-1])/self.th
                    # print('deltaloglike/th ='+str(toprint))
                    # print(f'{np.abs(deltlike/loglike[countf-1])}: th({self.th}; k={k})')

                if np.abs(deltlike/loglike[countf-1]) < self.th :
                    cont=False
                iteration+=1
            iteration=0

            if dl[countf] < mindl:
                self.bestpp = estpp
                self.bestmu = estmu
                self.bestcov = estcov
                self.bestk = k
                mindl = dl[countf]
                # for j in range(self.bestcov.shape[2]):
                    # print(f'best cov: {self.bestcov[:,:,j]}')

            if k>self.kmin:
                indminp = np.argmin(estpp)
                estmu = np.delete(estmu, indminp, axis=0)
                estcov = np.delete(estcov, indminp, axis=-1)
                estpp = np.delete(estpp, indminp, axis=-1)
                k=k-1
                estpp = estpp/np.sum(estpp)
                transitions2.append(countf)
                countf=countf+1
                indic = np.zeros((k,npoints))
                semi_indic = np.empty((k,y.shape[0]))
                for i in range(k):
                    min_eig = np.min(np.real(np.linalg.eigvals(estcov[:,:,i])))
                    if min_eig < 0:
                        estcov[:,:,i] -= 10*min_eig * np.eye(*estcov[:,:,i].shape)
                        # print('211\n')
                    semi_indic[i,:]=self._posterior_probability(y,estmu,estcov,i)
                    indic[i,:]=semi_indic[i,:]*estpp[:,i]

                if k != 1:
                    loglike.append(np.sum(np.log(np.finfo(np.float64).tiny+np.sum(indic,axis=0))))
                else:
                    loglike.append(np.sum(np.log(np.finfo(np.float64).tiny+indic)))

                dlength = -loglike[countf] + (nparsover2*np.sum(np.log(estpp))) + (nparsover2 + 0.5)*k*np.log(npoints)
                dl.append(dlength)

        #        countf=countf-1
                kappas.append(k)
            else:
                k_cont=False
        # for a in range(self.bestcov.shape[2]):
        #     print(f'end best cov: {self.bestcov[:,:,a]}')
        # for b in range(estcov.shape[2]):
        #     print(f'end cov: {estcov[:,:,b]}')
        return self

    def sample(self,sample):
        output=[]
        select_sample=np.random.multinomial(sample, self.bestpp[0])
        gmm=0
        for i in select_sample:
            for i in range(i):
                output.append(np.random.multivariate_normal(self.bestmu[gmm],np.swapaxes(self.bestcov,0,2)[gmm]))
            gmm+=1
        return np.array(output)

    def transform(self, X, y=None):
        y=np.array(X)
        semi_indic = np.empty((self.bestmu.shape[0],y.shape[0]))
        # for j in range(self.bestcov.shape[2]):
            # print(f'transform cov: {self.bestcov[:,:,j]}')
        for i in range(self.bestmu.shape[0]):
                    semi_indic[i,:]=self._posterior_probability(y,self.bestmu,self.bestcov,i)
        return semi_indic.T

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)

    def predict(self, X):
        return np.argmax(self.transform(X),axis=1)
    
    def classify_components(self, X):
        """
        分布している成分がどのクラスに分類されたかを格納する関数。

        Parameters:
            X (numpy.ndarray): 入力データ (N × D)

        Returns:
            list: 各クラス番号とそのクラスに属する成分の座標のリスト。
                形式: [(クラス番号, [座標リスト]), ...]
        """
        y = np.array(X)
        predictions = self.predict(X)  # 各データポイントのクラスを予測
        classified_components = []

        for class_id in range(self.bestmu.shape[0]):  # 各クラスについてループ
            # 指定クラスに属する成分の座標を取得
            class_points = y[predictions == class_id]
            # クラス番号とそのクラスに属する成分の座標を追加
            classified_components.append((class_id, class_points.tolist()))
        
        return classified_components
