# Report – Week 5: Gradients & Initialization

**Presenter:** \[Name]  

**Date:** \[DD.MM.YYYY] 

---
## Summary

---
## Discussion Notes

**Q:** Is there a difference between computing the gradient for the combined loss (like it is done in Pytorch) vs. computing it "point-wise"? \
**A:** No, since mathematically there is no difference whether you differentiate first or you average first.


**Q:** How would you determine that your parameters were initialized poorly? \
**A:** You either get an exploding gradient or vanishing gradient. An exploding gradient may be visible in the loss because the loss becomes very volatile. A vanishing gradient might make it look like the loss is not changing at all anymore.

**Q:** In theory, what would happen if we intialized all weights with identical values? \
**A:** If all weights were the same, all the gradients would also be identical, meaning you would not be able to distinguish between neurons and the model could not describe anything useful.

**Q:** Regarding the formula for optimal initialization weights, where did the 2 come from? \
**A:** The 2 originally came from the observation that the ReLU discards half the values for μ=0, meaning it also halves the variance from one layer to the next.

**Q:** What is the difference between static and dynamic computation graphs in ML frameworks (i.e. Tensorflow vs. Pytorch)? \
**A:** In general, Tensorflow is faster, while Pytorch is more transparent, which is why researchers often use Pytorch. 

**Q:** We heard that backpropagation saves a lot of computation when computing the gradients. Are there also costs/ disadvantages to this approach? \
**A:** Since backpropagation saves many intermediate values for its next steps, it is not particularly memory efficient. There are approaches to cope with this, like only caching the most important values and re-computing other ones to save space, but it is always a space-time-tradeoff.

**Q:** The book said that one of the reasons that initialization is a potential problem is because of floating point precision. In theory, what would happen if we had infinite precision/ compute? \
**A:** At this point, the question becomes "Can I find the global minimum from any starting point?", which should be possible with infinite compute.
