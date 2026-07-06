julia.chae2000
5 July, 2:21 pm
This is confusingly worded and I'm not sure comes across cleanly to someone who is reading the paper for the first time, possible reword?
Reply
julia.chae2000
5 July, 2:22 pm
edit
Changed: an obvious
to a possible
julia.chae2000
5 July, 2:25 pm
This figure seems to imply that ResNet-50 is linear probed, when it says below that it's fine-tuned. Which is true? I think it should be fine-tuned if that hasn't been run, and also update the figure.
Reply
julia.chae2000
5 July, 2:24 pm
what does structured morphological prompting mean here, that's for generation right? Is this structured prompting not also applied to the other GPT-Image synthetic generations? e.g., isnt' the original generative image pool identical, other than the filtering? This makes it seem like this type of generation/prompting is specific to the expert-calibrated selection probe.
Reply
julia.chae2000
5 July, 2:33 pm
Missing citations that should be added!

Fill-Up: Balancing Long-Tailed Data with Generative Models (past work that specifically focuses on long-tailed classification improvements with textual-inversion). Should definitely include this paper in related works, and potentially even as a baseline.

Personalized Representation from Personalized Generation - our work a few years ago that specifically focuses on vision representation improvement in extremely data-scarce regime using synthetic data!

Maybe it's worth noting for those two citations: often in a long-tailed regime, due to data-deficiency, fine-tuning methods like personalized generations (e..g, textual inversion, dream booth) are popular because using off-the-shelf models may not be very strong in this regime. Both of above works leverage those fine-tuned methods rather than using off-the-shelf generators.

Reply
julia.chae2000
5 July, 2:40 pm
Could we include a figure here of the three target species and why they might be difficult (e.g., show confusion species,)
Reply
julia.chae2000
5 July, 2:45 pm
Why was 200 selected when that isn't the amount of images needed to make the classes more balanced? Are there any analyses done on scaling this matched volume? (e.g., how do the results change if it's +50, +100, +200 vs +500, or complete fill-up?) Otherwise +200 seems arbitrary and it's unsure if it has been selected for good reason.
Reply
julia.chae2000
5 July, 2:44 pm
Is it trained from scratch or fine-tuned from ImageNet weights? The figure seems to imply linear probe, above section explains that it's fine-tuned and now it mentioned trained from scratch so it would be good to make sure that everything is consistent!
Reply
julia.chae2000
5 July, 2:50 pm
Can you justify why the generation was done in this way? For example, did you also test text-guided generation instead of reference-guided and find that it was better? Again, why 500 images per rare species instead of like complete fill-up to ~900 images total per species? I know that explorations were done around the prompt engineering, so would you be able to include how that structured template was ultimately selected and what else was tried?  
Reply
julia.chae2000
5 July, 2:53 pm
Maybe can group 4.3, 4.4 and 4.5 as "Improving Fidelity of Reference-Based Generative Images with Filtering" and have sub-subsections each for LLM-as-a-judge, BioCLIP centroid, and Expert-Calibrated Probe.
Reply
julia.chae2000
5 July, 2:55 pm
Could we try BioCLIP2 instead of bioCLIP?
Reply
julia.chae2000
5 July, 2:57 pm
This is a bit unclear, what does this mean?
Reply
julia.chae2000
5 July, 2:59 pm
I think this figure should be re-generated, as the figure titles and legends are very small and not possible to read. Also for the right figure, it's too small that it's not possible to see the shape differences.
Reply
julia.chae2000
5 July, 3:02 pm
I'm not sure the point here is super clear -- what is 0.1-0.2 referring to? is it real-image distance to its own centroid? is synthetic-to-centroid referring to real species centroid of its own centroid?
Reply
julia.chae2000
5 July, 3:03 pm
Are there any figures / visualizations of successful images and see if that matches with the correct human classification?
Reply
julia.chae2000
5 July, 3:06 pm
edit
Changed: deployed $200$-image selection
to selected $200$-image subset
julia.chae2000
5 July, 3:09 pm
Here, I get that relative to the other filters the expert filters retained more, but what is the upper limit / oracle of what it should have retrained? would it be 27/17/6? Could you make this clear?
julia.chae2000
5 July, 7:26 pm
Also, I think Fig 6b) is a bit confusing and I'm not sure if it's bringing the right point across. Instead of this, maybe you can show the comparison in filtering classification performance against the expert labels for the three filtering methods? LLM vs Centroid VS Probe? Because majority of the 200 selected images are not expert-labelled, I think the point here is a bit muddied.
Reply
julia.chae2000
5 July, 8:27 pm
I think for rigorous results, we are missing a few baselines! We should add a few simple classical long-tailed baselines such as oversampling, class-balanced loss, LDAM-DRW, balanced softmax, then simple data augmentation baselines such as randaugment, mixup, and then also try to mimic Fill-Up's training recipe (synthetic + real training, then real-only finetuning with balacned softmax) since it's a recipe that was published previously + known to work. Without these, we might receive a lot of questions on how these numbers compare!
Reply
julia.chae2000
5 July, 8:13 pm
I think the statistical rigor here by trying bootstraping for single split, multi seed AND five-fold CV is great, but I think it's a little bit too much to all add into main results, especially because the results contradict each other and don't support a single clear narrative. I know that's part of the finding (e.g., there isn't a single best method), but I think it's confusing to the reader. Also for the CV results, I'd make sure that there's no data bleeding between folds, especially with the synthetic data pool re-use across folds. You mention that the generative images are reference-based and cut and paste images are from training foreground data, so we need to make sure that only the synthetic data that was generated from a sample that ends up in the training split are used in CV-fold validation. Otherwise there's bleeding and the results are not rigorous!
julia.chae2000
5 July, 8:20 pm
My suggestion is to remove single split and five-fold CV and keep a single clean split of train/val/test without any bleeding (e.g., all images that CP were generated from and were used as references for reference-based generation should be in train, not in val/test) and run over 10 seeds.
