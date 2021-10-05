import numpy as np
from .langevin import langevin

class langevinNoiseSampler(langevin):
    '''
    Integrator class to integrate the diffusive dynamics of a Brownian particle following the Langevin equation,
    where the noise term and interaction terms in the velocity equation are sampled from an alternate data-based
    model. Takes same input as Langevin integrator with the addition of a noise sampler. The noise sampler
    takes certain input and outputs a corresponding noise term.
    '''

    def __init__(self, dt, stride, tfinal, noiseSampler, kBT=1, boxsize = None, boundary = 'periodic'):
        # inherit all methods from parent class
        super().__init__(dt, stride, tfinal, kBT, boxsize, boundary)
        self.noiseSampler = noiseSampler
        self.prevNoiseTerm = np.zeros(3)

    def getConditionedVars(self, particle):
        return (particle.nextPosition)
        #return np.concatenate((particle.nextPosition, particle.nextVelocity))

    def integrateO(self, particleList):
        '''Integrates velocity full time step given friction and noise term'''
        for particle in particleList:
            conditionedVars = self.getConditionedVars(particle)
            eta = self.kBT / particle.D # friction coefficient
            noiseTerm = self.noiseSampler.sample(conditionedVars)
            self.prevNoiseTerm = 1.0 * noiseTerm
            frictionTerm = np.exp(-self.dt * eta/ particle.mass) * particle.nextVelocity
            particle.nextVelocity = frictionTerm + noiseTerm/(particle.mass)
