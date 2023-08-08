import numpy as np
import sys
import deepRD
import deepRD.tools.particleTools as particleTools
from .diffusionIntegrator import diffusionIntegrator
from ..reactionModels import reservoir
from ..reactionIntegrators import tauleap

class smoluchowski(diffusionIntegrator):
    '''
    Integrator class to integrate the diffusive dynamics of a standard Brownian particle with a reservoir at r=R and
    a partially absorbing boundary at r=sigma with sigma <R
    '''

    def __init__(self, dt, stride, tfinal, D = 1.0, kappa = 1.0, sigma = 0.0, R = 1.0, deltar = 0.1, cR = 1.0,
                 equilibrationSteps = 0, tauleapSubsteps = 10):
        '''
        inherit all methods from parent class
        Boundary delimited by [sigma,R], reactive boundary at sigma, boundary in contact with reservoir at R,
        deltar is the width of boundary layer to interact with reservoir and cR the concentration of the reservoir.
        refreshTimeStep is the number of timesteps that the particleList is refreshed (due to deactivated particles).
        '''
        kBT = 1
        super().__init__(dt, stride, tfinal, kBT, None, None, equilibrationSteps)
        self.D = D
        self.kappa = kappa
        self.sigma = sigma
        self.R = R
        self.deltar = deltar
        self.cR = cR
        self.nR = None
        self.injectionRate = 0.0
        #self.kappaDiscrete = self.kappa/(4 * np.pi * self.sigma**2 * self.deltar)
        denom = (4 * np.pi * self.deltar) * (3 * self.sigma ** 2 + 3 * self.deltar * self.sigma + self.deltar ** 2)
        self.kappaDiscrete = 3 * self.kappa / denom
        self.reservoirModel = None

        self.tauleapSubsteps = tauleapSubsteps
        self.refreshTimeSteps = 50
        self.tauleapIntegrator = tauleap(self.dt/2.0)

    def setTauleapSubsteps(self, substeps):
        self.tauleapSubsteps = substeps

    def setIntrinsicReactionRate(self, kappa):
        self.kappa = kappa
        #self.kappaDiscrete = self.kappa/(4*np.pi*self.sigma**2*self.deltar) # Just first order
        denom = (4 * np.pi * self.deltar)*(3*self.sigma**2 + 3*self.deltar*self.sigma + self.deltar**2)
        self.kappaDiscrete = 3 * self.kappa/denom

    def setReservoirModel(self, reservoirConcentation):
        Rn = self.R - self.deltar/2.0
        perParticleJumpRate = (self.D / (self.deltar ** 2)) * (1 - self.deltar / Rn)
        reservoirVolume = 4 * np.pi * ((self.R + self.deltar)**3 - self.R**3)/3.0
        self.reservoirModel =  reservoir(perParticleJumpRate, reservoirVolume, reservoirConcentation)

    # def setInjectionRate(self, reservoirConcentation):
    #     Rn = self.R - self.deltar/2.0
    #     perParticleJumpRate = (self.D / (self.deltar ** 2)) * (1 - self.deltar / Rn)
    #     reservoirVolume = 4 * np.pi * ((self.R + self.deltar)**3 - self.R**3)/3.0
    #     self.nR = reservoirVolume * reservoirConcentation
    #     self.injectionRate = self.nR * perParticleJumpRate

    def injectParticles(self, particleList, deltat):
        # Count number of reactions by running a tau-leap approximation
        #numInjectedParticles = np.random.poisson(self.injectionRate * deltat)
        self.tauleapIntegrator.dt = deltat
        self.reservoirModel.X = np.array([0])
        self.reservoirModel.updatePropensities()
        X = self.tauleapIntegrator.integrateMany(self.reservoirModel, self.tauleapSubsteps)
        numInjectedParticles = np.int(X[0])
        for i in range(numInjectedParticles):
            position = particleTools.uniformShell(self.R - self.deltar, self.R)
            particle = deepRD.particle(position, D = self.D)
            particleList.addParticle(particle)

    def diffuseParticles(self, particleList, deltat):
        self.calculateForceField(particleList)
        for i, particle in enumerate(particleList):
            if particle.active:
                sigma = np.sqrt(2 * deltat * particle.D)
                force = self.forceField[i]
                # Enforce reflective boundary by rejection sampling on the diffusion step
                rr = 0.0
                while (rr <= self.sigma):
                    nextPosition = particle.position + force * deltat * particle.D / self.kBT + \
                                   sigma * np.random.normal(0., 1, particle.dimension)
                    rr = np.linalg.norm(nextPosition)
                particle.nextPosition = nextPosition
                # Enforce absorbing boundary at reservoir interface
                if rr > self.R:
                    particleList.deactivateParticle(i)

    # # Perhaps this routine is not exact because then the particle should react in the expected lagtime if t remains
    # def partiallyAbsorbingReactionBoundaryOld(self, particleList, deltat):
    #     for i, particle in enumerate(particleList):
    #         if particle.active:
    #             rr = np.linalg.norm(particle.nextPosition)
    #             if rr <= self.sigma + self.deltar:
    #                 # Gillespie time of reaction check
    #                 r1 = np.random.rand()
    #                 lagtime = np.log(1.0 / r1) / self.kappaDiscrete
    #                 if lagtime <= deltat:
    #                     particleList.deactivateParticle(i)

    def partiallyAbsorbingReactionBoundary(self, particleList, deltat):
        reactProb = 1.0 - np.exp(-1.0 * self.kappaDiscrete * deltat)
        for i, particle in enumerate(particleList):
            if particle.active:
                rr = np.linalg.norm(particle.nextPosition)
                if rr <= self.sigma + self.deltar:
                    # Exponential reaction event sampling
                    r1 = np.random.rand()
                    if r1 <= reactProb:
                        particleList.deactivateParticle(i)


    # def enforceBoundary(self, particleList, currentOrNextOverride = None):
    #     for i, particle in enumerate(particleList):
    #         if particle.active:
    #             rr = np.linalg.norm(particle.nextPosition)
    #             # If particle reached reactive boundary, either react or reflect
    #             if rr <= self.sigma:
    #                 # Reflective BC
    #                 dr = self.sigma - rr
    #                 particle.nextPosition = particleTools.uniformSphere(self.sigma + dr)
    #                 ## Gillespie time of reaction check
    #                 #r1 = np.random.rand()
    #                 #lagtime = np.log(1.0 / r1) / self.kappa
    #                 #if lagtime <= self.dt:
    #                 #    particleList.deactivateParticle(i)
    #                 #else:
    #                     ## Reflection BC
    #                     #dr = self.sigma - rr
    #                     #particle.nextPosition = particleTools.uniformShell(self.sigma, self.sigma + dr)
    #                     ## Reflection BC 2
    #                     #dr = self.sigma - rr
    #                     #particle.nextPosition = particleTools.uniformSphere(self.sigma + dr)
    #                     ## Reflection BC 3
    #                     #particle.nextPosition = particleTools.uniformShell(self.sigma, self.sigma + self.deltar)
    #                     # Uniform on sphere
    #                     #particle.nextPosition = particleTools.uniformSphere(self.sigma)
    #             # Deactivate particles leaving into Reservoir (r>R)
    #             elif rr > self.R:
    #                 particleList.deactivateParticle(i)

    def integrateOne(self, particleList):

        # Injection/reaction/diffusion splitting algorithm
        self.injectParticles(particleList, self.dt/2.0)

        self.partiallyAbsorbingReactionBoundary(particleList, self.dt/2.0)

        self.diffuseParticles(particleList, self.dt)

        self.partiallyAbsorbingReactionBoundary(particleList, self.dt/2.0)

        self.injectParticles(particleList, self.dt/2.0)

        particleList.updatePositions()

    def propagate(self, particleList, showProgress = False):
        if self.firstRun:
            # Set injectionRate, assumes all particles have same diffusion coefficient
            if len(particleList) > 1 and self.D != particleList[0].D:
                raise NotImplementedError("Please check diffusion coefficient of integrator matches particles")
            self.setReservoirModel(self.cR)
            self.prepareSimulation(particleList)
            self.tauleapIntegrator.setSimulationParameters(self.dt/2.0)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleList)
            if i%self.refreshTimeSteps == 0:
                particleList.removeInactiveParticles()
        time = 0.0
        xTraj = [particleList.activePositions]
        tTraj = [time]
        for i in range(self.timesteps):
            self.integrateOne(particleList)
            if i%self.refreshTimeSteps == 0 or i == self.timesteps - 1:
                particleList.removeInactiveParticles()
                #sys.stdout.write(str(int(100*(i+1)/self.timesteps)) + '% Number of particles: ' + str(particleList.countParticles()) + "\r")
            # Update variables
            time = time + self.dt
            if i % self.stride == 0 and i > 0:
                xTraj.append(particleList.activePositions)
                tTraj.append(time)
            if showProgress and (i % 50 == 0):
                # Print integration percentage
                sys.stdout.write("Percentage complete " + str(round(100 * time/ self.tfinal, 1)) + "% " + "\r")
        if showProgress:
            sys.stdout.write("Percentage complete 100% \r")
        return np.array(tTraj), xTraj

class smoluchowskiAndBimolecularReactions(smoluchowski):
    '''
        Integrator class analogous to Smoluchowski for B particles but with additional A particle sin the domain
        that can undergo reactions like A+B->0 (WE WANT THIS REACTION OR A+B-> 2A and A->0).
        '''

    def __init__(self, dt, stride, tfinal, D = 1.0, kappa = 1.0, sigma = 0.0, R = 1.0, deltar = 0.1, cR = 1.0,
             reactionDistance = None, reactionRate = 0.0, equilibrationSteps = 0, tauleapSubsteps = 10):

        super().__init__(dt, stride, tfinal, D, kappa, sigma, R, deltar, cR, equilibrationSteps, tauleapSubsteps)
        self.reactionDistance = reactionDistance
        self.reactionRate = reactionRate

    def propagate(self, particleListA, particleListB, showProgress=False):
        if self.firstRun:
            # Set injectionRate, assumes all particles of same type have same diffusion coefficient
            if len(particleListA) > 1 and self.D != particleListA[0].D:
                raise NotImplementedError("Please check diffusion coefficient of integrator matches particles")
            if len(particleListB) > 1 and self.D != particleListB[0].D:
                raise NotImplementedError("Please check diffusion coefficient of integrator matches particles")
            self.setReservoirModel(self.cR)
            self.prepareSimulation(particleListB)
            self.tauleapIntegrator.setSimulationParameters(self.dt / 2.0)
        # Equilbration runs
        for i in range(self.equilibrationSteps):
            self.integrateOne(particleListA, particleListB)
            if i % self.refreshTimeSteps == 0:
                particleListA.removeInactiveParticles()
                particleListB.removeInactiveParticles()
        time = 0.0
        xTrajA = [particleListA.activePositions]
        xTrajB = [particleListB.activePositions]
        tTraj = [time]
        for i in range(self.timesteps):
            self.integrateOne(particleListA, particleListB)
            if i % self.refreshTimeSteps == 0 or i == self.timesteps - 1:
                particleListA.removeInactiveParticles()
                particleListB.removeInactiveParticles()
                # sys.stdout.write(str(int(100*(i+1)/self.timesteps)) + '% Number of particles: ' + str(particleList.countParticles()) + "\r")
            # Update variables
            time = time + self.dt
            if i % self.stride == 0 and i > 0:
                xTrajA.append(particleListA.activePositions)
                xTrajB.append(particleListB.activePositions)
                tTraj.append(time)
            if showProgress and (i % 50 == 0):
                # Print integration percentage
                sys.stdout.write("Percentage complete " + str(round(100 * time / self.tfinal, 1)) + "% " + "\r")
        if showProgress:
            sys.stdout.write("Percentage complete 100% \r")
        return np.array(tTraj), xTrajA, xTrajB


    def bimolecularReactions(self, particleListA, particleListB, deltat):
        reactionProbability = 1 - np.exp(self.reactionRate * deltat)
        for i, particleA in enumerate(particleListA):
            if particleA.active:
                for j, particleB in enumerate(particleListB):
                    if particleB.active:
                        distance = np.linal.norm(particleA.position - particleB.position)
                        if distance <= self.reactionDistance:
                            r1 = np.random.rand()
                            if r1 <= reactionProbability:
                                particleListB.deactivateParticle(j)
                                break
                # NEED to add reactions at boundary with reservoir here


    def integrateOne(self, particleListA, particleListB):

        # Injection/reaction/diffusion splitting algorithm
        self.injectParticles(particleListB, self.dt / 2.0)

        self.partiallyAbsorbingReactionBoundary(particleListB, self.dt / 2.0)

        self.bimolecularReactions(particleListA,particleListB, self.dt/2)

        self.diffuseParticles(particleListA, self.dt)

        self.diffuseParticles(particleListB, self.dt)

        self.bimolecularReactions(particleListA, particleListB, self.dt / 2)

        self.partiallyAbsorbingReactionBoundary(particleListB, self.dt / 2.0)

        self.injectParticles(particleListB, self.dt / 2.0)

        particleListA.updatePositions()

        particleListB.updatePositions()
