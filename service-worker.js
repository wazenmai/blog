if(!self.define){let e,s={};const i=(i,a)=>(i=new URL(i+".js",a).href,s[i]||new Promise((s=>{if("document"in self){const e=document.createElement("script");e.src=i,e.onload=s,document.head.appendChild(e)}else e=i,importScripts(i),s()})).then((()=>{let e=s[i];if(!e)throw new Error(`Module ${i} didn’t register its module`);return e})));self.define=(a,r)=>{const n=e||("document"in self?document.currentScript.src:"")||location.href;if(s[n])return;let o={};const c=e=>i(e,n),t={module:{uri:n},exports:o,require:c};s[n]=Promise.all(a.map((e=>t[e]||c(e)))).then((e=>(r(...e),o)))}}define(["./workbox-d249b2c8"],(function(e){"use strict";self.addEventListener("message",(e=>{e.data&&"SKIP_WAITING"===e.data.type&&self.skipWaiting()})),e.precacheAndRoute([{url:"about/index.html",revision:"b5aa17701d1acb1cde0ea008e2cd9f4f"},{url:"archives/2021/08/index.html",revision:"51586f1f3bb99becad797976f51d6871"},{url:"archives/2021/index.html",revision:"4b9160e4832bf29461d259f8a454005e"},{url:"archives/2022/11/index.html",revision:"6f02026dcbdb1c3a9485ed604083992f"},{url:"archives/2022/index.html",revision:"3921f9c059c8c01b509378544df17cdf"},{url:"archives/index.html",revision:"a87e73e4954304c8da4bd231218b1c3d"},{url:"assets/css/style.css",revision:"8c3757344b6b13071c5d05c4af8c7b7c"},{url:"assets/posts/Arc-Browser/arc-demo.png",revision:"6827be6be2a7fad0b8b5d050365e0e33"},{url:"assets/posts/Arc-Browser/arc-sidebar.png",revision:"c3ac1c713fa8a7f339d49d07215a8594"},{url:"assets/posts/Arc-Browser/library.gif",revision:"1e9b61b1cf3afbf9f49c1c63c0212fd5"},{url:"assets/posts/Arc-Browser/test.gif",revision:"c626814c61077cffbb3d130e8ebf7df5"},{url:"assets/posts/Arc-Browser/wazenmai_Arc.png",revision:"d9f9b56cf3a643d6c44b175d0d0fe408"},{url:"assets/posts/hello-world/asexiaul.png",revision:"c2c2088ab1cbd774faa7af7611c3ba06"},{url:"assets/posts/MIDI-BERT/midibert.png",revision:"5da52842197c7fb6da7698f9d5def800"},{url:"assets/posts/MIDI-BERT/top_img.png",revision:"513d0966e4e502a1ec7063ea422c320a"},{url:"assets/posts/ProDA/ablation.png",revision:"4bd323059dad34f5d601aa39ba4c7ea0"},{url:"assets/posts/ProDA/GTA5.png",revision:"b946f496f744b1bf7670661981a3558c"},{url:"assets/posts/ProDA/Issue.png",revision:"d650457b4aa98503367cfb813b917937"},{url:"assets/posts/ProDA/model1.png",revision:"156a0aece175d4a09006ea0dd10156af"},{url:"assets/posts/ProDA/model2.png",revision:"ceb6d0fc739af1aecea1716e3bb3d7ef"},{url:"assets/posts/ProDA/model3.png",revision:"bf06086e0193abf137ac52d4f3703c08"},{url:"assets/posts/ProDA/ProDA.png",revision:"352291ff29e29f6cd05e13539e6e9209"},{url:"assets/posts/ProDA/prototype.png",revision:"c1de210fabeab132d328db5af38ed34e"},{url:"assets/posts/ProDA/Synthia.png",revision:"dd83f8b2ce96753e5d587fe80bcb7ea8"},{url:"assets/posts/ProDA/umap.png",revision:"1eafbff90f921f574e0fc93d1624b288"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/CC_NTHU.png",revision:"102a9e77e939963c869469d5f180850e"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/GTA5_CC.png",revision:"b1bccc58571d2cebbe4f676b6d7368c3"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/model.png",revision:"7f06b8e84b65d8e4e3bc0764805a2c17"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/Syn_CC.png",revision:"0721bb6be9270b3f36d999d4d0d5db22"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/toy_example.png",revision:"160e540db03d05ba99505cf028cde877"},{url:"assets/posts/Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/toy_problem.png",revision:"2dc876fc57cb0b1d1a7b231b0aab657d"},{url:"assets/profile.jpg",revision:"fc93d378e50a7fde556e4ff2b02589c2"},{url:"assets/top_img.png",revision:"e438f889cebec6a1d226a8aa7d4bffef"},{url:"categories/index.html",revision:"5b1669e85d2c4316b491ec9f809af721"},{url:"categories/Machine-Learning/index.html",revision:"b411f5153424ee0738a6f0c1f9c844ba"},{url:"categories/好物推薦/index.html",revision:"ad9d2ef2df30ca380044a3749e9d8151"},{url:"css/index.css",revision:"37c9252ac2fb22d06f5768728cd9190d"},{url:"css/var.css",revision:"d41d8cd98f00b204e9800998ecf8427e"},{url:"img/404.jpg",revision:"4ef3cfb882b6dd4128da4c8745e9a507"},{url:"img/favicon.png",revision:"7a8c47cb5a2149c1a1af21e90ecd9ca7"},{url:"img/friend_404.gif",revision:"68af0be9d22722e74665ef44dd532ba8"},{url:"index.html",revision:"c0017b35a22afeb15dc6dcb05280fad3"},{url:"js/main.js",revision:"b28662bf81abe2838d20faf2980f0034"},{url:"js/search/algolia.js",revision:"308de383b932f321e04751fd1f79cffb"},{url:"js/search/local-search.js",revision:"149fcc60c1de0a818e111978d01cbd87"},{url:"js/tw_cn.js",revision:"58cb9443474ea5d84a1c4ac806c8b4b7"},{url:"js/utils.js",revision:"24971090b8b1bd5d3f538d414e270fd3"},{url:"music/index.html",revision:"8bc4c2a29e6d49e091c2fe19395bbd22"},{url:"tags/Recommandation/index.html",revision:"b3567c39322c85d295b51969ac5642ed"},{url:"tags/Semantic-Segmentation/index.html",revision:"7006c83dd35d2313c9cf00ae5f751364"},{url:"tags/UDA/index.html",revision:"56bf131467659caeb4456a41fd721fd0"},{url:"好物推薦-Arc-Browser/index.html",revision:"0010d50deb4448c1e3cf258a00ec0575"},{url:"論文筆記-ProDA-Prototypical-Pseudo-Label-Denoising-and-Target-Structure-Learning-for-Domain-Adaptive-Semantic-Segmentation/index.html",revision:"320f29689da7207fa52ba4388f93635b"},{url:"論文筆記-Uncertainty-Reduction-for-Model-Adaptation-in-Semantic-Segmentation/index.html",revision:"d8c767d45f1fa6a1b87765adb93c591c"}],{})}));
//# sourceMappingURL=service-worker.js.map
