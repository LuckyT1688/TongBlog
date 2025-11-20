import React, { useEffect, useRef, useState } from "react";

interface SearchResult {
  item: {
    data: {
      title: string;
      description?: string;
      createdAt?: string;
    };
    url: string;
  };
}

declare global {
  interface Window {
    pagefind: any;
  }
}

const SearchPage = () => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [inputVal, setInputVal] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isPagefindLoaded, setIsPagefindLoaded] = useState(false);

  const handleChange = (e: React.FormEvent<HTMLInputElement>) => {
    setInputVal(e.currentTarget.value);
  };

  // Load Pagefind
  useEffect(() => {
    const loadPagefind = async () => {
      if (typeof window !== "undefined") {
        try {
          // @ts-ignore
          window.pagefind = await import(/* @vite-ignore */ "/pagefind/pagefind.js");
          await window.pagefind.init();
          setIsPagefindLoaded(true);
        } catch (e) {
          console.warn("Pagefind failed to load (this is expected in dev mode unless configured):", e);
        }
      }
    };
    loadPagefind();
  }, []);

  useEffect(() => {
    const searchUrl = new URLSearchParams(window.location.search);
    const searchStr = searchUrl.get("q");
    if (searchStr) setInputVal(searchStr);

    requestAnimationFrame(() => {
      if (inputRef.current) {
        inputRef.current.selectionStart = inputRef.current.selectionEnd =
          searchStr?.length || 0;
      }
    });
  }, []);

  useEffect(() => {
    const performSearch = async () => {
      if (inputVal.length >= 2 && isPagefindLoaded && window.pagefind) {
        const search = await window.pagefind.search(inputVal);
        // Get top 10 results
        const results = await Promise.all(search.results.slice(0, 10).map((r: any) => r.data()));
        
        const mappedResults = results.map((data: any) => ({
          item: {
            data: {
              title: data.meta.title,
              description: data.excerpt, // Pagefind returns HTML excerpt
              createdAt: data.meta.date,
            },
            url: data.url,
          },
        }));
        
        setSearchResults(mappedResults);

        const searchParams = new URLSearchParams(window.location.search);
        searchParams.set("q", inputVal);
        const newRelativePathQuery =
          window.location.pathname + "?" + searchParams.toString();
        history.pushState(null, "", newRelativePathQuery);
      } else {
        if (inputVal.length < 2) {
            history.pushState(null, "", window.location.pathname);
            setSearchResults([]);
        }
      }
    };

    const timeoutId = setTimeout(performSearch, 300);
    return () => clearTimeout(timeoutId);
  }, [inputVal, isPagefindLoaded]);

  return (
    <section className="">
      <div className="container px-3 lg:px-8">
        <div className="row mb-10 justify-center">
          <div className="col-10 lg:col-8 px-0">
            <div className="flex flex-nowrap">
              <input
                className="w-full glass rounded-[35px] p-6 text-txt-p placeholder:text-txt-light dark:placeholder:text-darkmode-txt-light focus:border-darkmode-border focus:ring-transparent dark:text-darkmode-txt-light intersect:animate-fadeDown opacity-0 intersect-no-queue"
                placeholder={isPagefindLoaded ? "搜点什么..." : "搜索功能仅在构建后可用"}
                type="search"
                name="search"
                value={inputVal}
                onChange={handleChange}
                autoComplete="off"
                autoFocus
                ref={inputRef}
                disabled={!isPagefindLoaded}
              />
            </div>
          </div>
        </div>
        <div className="row">
          {searchResults?.length < 1 ? (
            <div className="col-10 lg:col-8 mx-auto p-2 text-center glass rounded-[35px] intersect:animate-fadeUp opacity-0">
              <p id="no-result">
                {!isPagefindLoaded 
                  ? "开发模式下搜索不可用，请 build 后测试" 
                  : inputVal.length < 1
                  ? "“嗖”的一下，就搜出来了！"
                  : inputVal.length < 2
                  ? "请输入2个以上字符"
                  : "我没找到呢，试试其他关键词"}
              </p>
            </div>
          ) : (
            searchResults?.map(({ item }, index) => (
                <div className="py-2 px-0" key={`search-${index}`}>
                  <div className="h-full glass col-10 lg:col-8 mx-auto rounded-[35px] p-6 intersect:animate-fade opacity-0">
                    <h4 className="mb-2">
                      <a href={item.url}>
                      {item.item.data.title}
                      </a>
                    </h4>
                  { item.item.data.description && (
                    <p className="" dangerouslySetInnerHTML={{ __html: item.item.data.description }} />
                    )}
                  {item.item.data.createdAt && (
                    <p className="text-txt-light dark:text-darkmode-txt-light">
                      {new Date(item.item.data.createdAt).toLocaleDateString()}
                    </p>
                  )}
                  </div>
                </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
};

export default SearchPage;
