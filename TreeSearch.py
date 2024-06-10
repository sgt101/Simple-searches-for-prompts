
#import python utils
import random
import threading
import json
from typing import Dict, Optional
from time import sleep
#from tqdm import tqdm_notebook as tqdm
from perc import Perc
import datetime

#import libraries for application
import dsp
import dspy
from dspy.primitives import Example

from dspy.teleprompt import Teleprompter
from dspy.teleprompt import LabeledFewShot

class TreeSearch(Teleprompter):
    
    _training_set = None
    _metric = None
    _model = None
    _pruning_threshold=0.0
    _pruning_delay= 0.0 
    _max_labeled_demos=0
    _sparsity_demos= 0.0
    _breadth = 0
    _max_rounds=50
    _model = None
    _system_prompt = None
    
    
    def __init__(
        self,
        training_set=None, 
        metric=None,
        model=None,
        pruning_threshold=0.03,
        pruning_delay=20,
        max_labeled_demos=16,
        sparsity_demos=0.5,
        breadth = 10,
        max_rounds=5
        ):
        """
        A tree search type optimiser for prompts. 
        Take a dict of demonstrations (dspy examples), create a set of trials (default 10)

        Parameters
        ----------
        metric: Callable
            A function that compares an expected value and predicted value, outputting the result of that comparison. 
        training_set: dict
            The examples that can be selected to form part of the prompt
        pruning_threshold: optional float, default 0
            Decides when to prune a search, if set to 0 then will prune when the mean of the current metric results * remaining 
            examples to be evaluated < best result so far (so the expectation is that this trial will fail). Default is 0.03 so 
            pruning will happen if the best trial so far is more than 3% above the current expectation for this candidate. 
            Essentially we are saying that we don't think a step will ever improve by 3% during the last part of the evaluation 
        pruning_delay: optional float, default 0.10 
            Decides how many examples to evaluate before trying to prune. 0.1 implies that 10% of the training set will be 
            evaluated before the first check, this gives some time for the estimate of the quality of the branch to settle. 
        max_labeled_demos: int, default 16
            Maximum number of labeled demonstrations to include.
        breadth: int, default 10 
            number of different candidate branches to evaluate at each step (the breadth of the search) there will be the same number of seeds as breadth 
        sparsity_demos: float, default 0.5
            How heavily packed the demonstration vector should be at the start of the search. 0.5 implies that the mean number of demonstrations in the program should be 1/2 of the maximum number 
        max_rounds: int, default 5
            Number of iterations to attempt generating the required set of examples. If after `max_rounds`, the search ends.
        model: the model selected (dspy object)
        """
        self._training_set = training_set
        self._metric = metric
        self._model = model
        self._pruning_threshold=pruning_threshold
        self._pruning_delay= pruning_delay 
        self._max_labeled_demos=max_labeled_demos
        self._sparsity_demos= sparsity_demos
        self._breadth = breadth
        self._max_rounds=max_rounds
        self._model = model
        self._system_prompt = "You are a software document writer. The following examples are templates for you to use, then there will be a java function for you to document using the template styles "
    

    
    def stepSearch (self,seeds, fitness, highest_so_far, highest_fitness):
        """
        Given a set of seeds ( max_labelled_demos * breadth), and their fitness (breadth) create a new set of candidate demon for evalation. If fitness is recorded as 0 that means that the example should be discarded because it was pruned in the evaluation phase. The highest rated candidate should be used to fill any pruned examples.
        When a set of examples is complete (pruned removed and backfilled) then each of the remaining candidates should be updated to change them by one step. 
            (TODO)
            if the candidate is sparser than the expectation then 
            - add another demonstration in
            (TODO)
            if the candidate is not sparse then 
            - delete one demonstration and leave the slot empty 
            if the candidate has the expected density then 
            - change one demonstration for another one selected at random 

        Args:
            seeds (List): An list of candidate example sets. Each candidate is an array of indexes to the training set
            fitness (List): An list of fitness values range -1.0->1.0 0.0 or below means that this is pruned.
            highest_so_far (List): the candidate description that has scored highest so far
            highest_fitness (_type_): _description_
        """
        for n in range (self._breadth):
            if fitness[n]==0.0: #pruned
                seeds[n]=highest_so_far # this problem is too hard to exhaust the space, so optimise on the best
            seed=seeds[n] #search on it regardless
            x = random.randrange(len(seed))
            seed[x]=random.randrange(len(self._training_set))
        return (seeds)


    def initaliseSeeds(self):
        """
        Create a new randomly initialised set of candidates returns a set of seeds randomly initialised using the parameters _sparsity_demos and _breadth
        """
        seeds=[]
        for n in range (self._breadth):
            seed=[]
            for m in range (self._max_labeled_demos):
                item = random.randrange(len(self._training_set))
                if (random.random()>(1.0-self._sparsity_demos)): # for % of the time leave the slot empty 
                    seed.append(item)
            seeds.append(seed)
        return (seeds)
            

    def evaluateCandidate(self, prompt,highest_fitness,evaluation_set):
        """
        Evaluate a prompt has been generated from a seed.
        _metric and _model need to be set for this to work (obviously)
        
        This is where the pruning needs to happen - the evaluation is very expensive so limiting useless calls is important. _pruning_threshold and _pruning_delay are 
        Args:
            prompt (String) : the rendered prompt that is to be tested 
            target (String) : the ideal/provided string that should be generated
            highest_fitness (Float) : the highest_fitness value yeilded by any Candidate so far 
        """
        count=1
        total_fitness=0.0
        for example in Perc (evaluation_set):
                #unfortunately responses are inconsistent - need to flatten to allow different models 
                #this is for llama3 
                if (random.random()<0.05):break # just for testing 
                
                response = self._model.request(prompt)["choices"][0]['message']['content']
                example_fitness = self._metric (response, example.answer)
                total_fitness = total_fitness + example_fitness
                # if you've done enough examples to have a fair estimate of the quality of this one 
                # then check to see if it's on track to get close to the highest_fitness - a threshold factor
                # so we're not demanding that the estimate is that it'll be the fittest, but we are 
                # saying that it should be within x% of the fittest. 
                if (count > self._pruning_delay) & (total_fitness/float(count) < highest_fitness - self._pruning_threshold):
                    total_fitness = 0.0 
                    break 
                count=count+1
        return (total_fitness/float(count))
    


        
    def renderPrompt(self,prompt, candidate, evaluation_set=_training_set):
        """look up the elements in the candidate and create a prompt that uses them. 
        Args:
            prompt (String ): intially _system_prompt 
            candidate (List): an list of integers that each reference a member of the evaluation_set
            evaluation_set (List): examples to be used to do evaluation, defaults to training_set, but could be test_set 
        """
        for n in range (len(candidate)):
            seed = candidate[n] #get the right candidate
            prompt = prompt + "example code is: " + evaluation_set[candidate[n]].question + "example documentation is: " + evaluation_set[candidate[n]].answer +"\n"
        return (prompt)
    
           
    def evaluateAll(self,seeds, highest_so_far, highest_fitness,evaluation_set):
        """
        estimate the fitness of the seeds 
        this is likely to be very computationally expensive so needs to be hard pruned to save time 
        """
        candidate_fitness =[]
        for n in Perc(range(self._breadth)):
            prompt = self.renderPrompt(self._system_prompt,seeds[n], evaluation_set)
            fitness = self.evaluateCandidate(prompt, highest_fitness,evaluation_set)
            candidate_fitness.append(fitness)
            if (fitness>highest_fitness) : 
                highest_so_far = seeds[n]
                highest_fitness = fitness
        return (candidate_fitness, highest_so_far, highest_fitness)
            
            
    def allPruned(self,fitness):
        """
            check to see if all the fitnesses are 0.0 
        """         
        pruned = True
        for n in range(len(fitness)):
            if (fitness[n]>0.0):
                pruned = False
                break
        return (pruned)


      
    def save (self,highest_so_far, highest_fitness, seeds, iteration, filename):
        
        #make a list so that we can dump it as json 
        to_make_json = []
        
        to_make_json.append(["highest_so_far", highest_so_far])
        to_make_json.append(["highest_fitness", highest_fitness])
        to_make_json.append(["seeds",seeds])
        to_make_json.append(["iteration",iteration])
        to_make_json.append(["pruning_threshold",self._pruning_threshold])
        to_make_json.append(["pruning_delay",self._pruning_delay])
        to_make_json.append(["sparsity_demos",self._sparsity_demos])
        to_make_json.append(["breadth",self._breadth])
        to_make_json.append(["max_rounds", self._max_rounds])
        file = open(filename, 'w')
        file.write (json.dumps(to_make_json))
        
        file.close()
        
        

        
    def testSave(self):
        """
        save a dummy session, note, you could do this to save a consistent seed and then
        test differnet parameters. 
        """
        seeds = self.initaliseSeeds()
        highest_so_far = seeds[1]
        highest_fitness = random.random()
        self.save(highest_so_far, highest_fitness, seeds, 10,  "test_save.json")
        
    
    def loadAndRun(filename, maxRounds):
        file = open(filename, 'r')
        for line in file:
            print(line)
        #return (highest_so_far, highest_fitness, seeds)
    
        
    def search (self):
        """ do a search of possible prompt completions using the training set that's been supplied during initialisation (_training_set) and the fitness function that has also been intialsied (_metric)
        The search is for _breadth candidates over _max_rounds of iterations. 
        if a candidate is _pruning_threshold below the best candidate then it will be removed and the best candidate will be used to create a new seed for further searching (so a branch is collapsed in the beam)

        Returns:
            highest_so_far (List): an list containing the indexes of the _training_set items that were in the best prompt 
            highest_fitness (Float) : the evaluated fitness for the best candidate discovered in the search 
            seeds (List): the final set of beam candidates
        """

        seeds = self.initaliseSeeds()
        highest_so_far = seeds[0]
        highest_fitness = 0.01
        count = 0 
        fitness=[0.01]*self._breadth #initialise the fitness list with low values
        
        for count in Perc(range(self._max_rounds)):
            seeds = self.stepSearch(seeds,fitness, highest_so_far, highest_fitness)
            fitness, highest_so_far, highest_fitness = self.evaluateAll(seeds, highest_so_far, highest_fitness,self._training_set)
            filename = "logs" + str(datetime.datetime.now()) + str(count) 
            self.save(highest_so_far,highest_fitness,seeds, count,filename)
            print ("round " + str(count) + "fitness = " +  str(highest_fitness))
            if (self.allPruned(fitness)): 
                #print ("all pruned")
                break #stop the loop if everything is pruned and return   
        #if (count == self._max_rounds):
            #print ("full run")
        return (highest_so_far, highest_fitness, seeds)
    

    highest_so_far=[]
    highest_fitness = 0.0
  
        
        
    def compile(self,metric=None, training_set = None): 
        if (metric != None):
            self._metric = metric
        if (training_set != None):
            self._training_set = training_set  
        if (self._metric==None):
            print ("you must provide a fitness function as the metric parameter")
            return()
        if (self._training_set==None):
            print ("you need to provide a training set to search")
            return()
        highest_so_far, highest_fitness, final_candidates = self.search()
        print (self.renderPrompt(self._system_prompt,highest_so_far))
        self.highest_so_far = highest_so_far
        self.highest_fitness = highest_fitness
        
        return (highest_so_far, highest_fitness)
    

    




